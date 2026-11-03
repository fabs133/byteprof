"""Boundary flags from the dolma2 tokenization of a byte string (Task B).

Bolmo's patches imitate the subword tokenizer of OLMo 2 (dolma2). `token_spans` gives the exact byte span of
every token of a text: the tokenizer is byte-level BPE, so a token string maps back to bytes through the
GPT-2 byte-to-unicode table, and a token may end inside a multi-byte character. A flag sits on the last byte
of a token (the convention of `corpus.segment_with_tokenizer` and of the recorded atoms).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def default_tokenizer_json() -> Path:
    return next((HF_CACHE / "models--allenai--OLMo-2-0425-1B" / "snapshots").glob("*/tokenizer.json"))


@lru_cache(maxsize=1)
def _unicode_to_byte() -> dict[str, int]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class Dolma2:
    """Wraps the `tokenizers` tokenizer; `token_spans(data)` -> [(start, end)] byte spans (end exclusive)."""

    def __init__(self, path: str | Path | None = None):
        from tokenizers import Tokenizer

        self.tk = Tokenizer.from_file(str(path or default_tokenizer_json()))
        u2b = _unicode_to_byte()
        self.tok_len: dict[int, int] = {}
        vocab = self.tk.get_vocab()
        for tok, i in vocab.items():
            if all(ch in u2b for ch in tok):
                self.tok_len[i] = len(tok)

    def id_bytes(self, i: int) -> bytes:
        """The bytes of token `i` (special tokens: empty)."""
        u2b = _unicode_to_byte()
        tok = self.tk.id_to_token(i)
        return bytes(u2b[ch] for ch in tok) if tok is not None and all(ch in u2b for ch in tok) else b""

    def encode_ids(self, text: str) -> list[int]:
        return self.tk.encode(text, add_special_tokens=False).ids

    def token_spans(self, data: bytes) -> tuple[list[tuple[int, int]], int]:
        """Spans of the tokens of `data`, and the number of bytes that were tokenized.

        Trailing bytes that are an incomplete UTF-8 sequence are dropped from the tokenized text (they cannot
        be tokenized yet); the second value tells how many bytes of `data` the spans cover."""
        text = data.decode("utf-8", errors="ignore")
        used = len(text.encode("utf-8"))
        if used != len(data):  # invalid bytes inside the data would shift the offsets: require a clean prefix
            used = _valid_prefix(data)
            text = data[:used].decode("utf-8")
        ids = self.encode_ids(text)
        spans: list[tuple[int, int]] = []
        pos = 0
        for i in ids:
            n = self.tok_len[i]
            spans.append((pos, pos + n))
            pos += n
        assert pos == used, (pos, used)
        return spans, used


def _valid_prefix(data: bytes) -> int:
    try:
        data.decode("utf-8")
        return len(data)
    except UnicodeDecodeError as e:
        return e.start


def spans_to_flags(spans: Sequence[tuple[int, int]], n: int) -> list[int]:
    f = [0] * n
    for _, e in spans:
        if 0 < e <= n:
            f[e - 1] = 1
    return f


def compare_flags(bolmo: Sequence[int], ref_spans: Sequence[tuple[int, int]], first: int, covered: int) -> dict:
    """Agreement of Bolmo's flags (one per byte position of the text) with the token spans.

    Evaluated positions: `first` <= p < `end_eval`, where `end_eval` is the end of the second-to-last token
    (the final token may be incomplete and is excluded, as is anything after the tokenized prefix). Patches:
    the Bolmo patches that end in the evaluated region (a patch starts after the previous flag, anywhere in the
    text) against the tokens; a patch is exact if a token has the same start and end."""
    if len(ref_spans) < 2:
        return {"positions": 0, "agree": 0, "patches": 0, "patches_exact": 0, "diffs": []}
    end_eval = min(ref_spans[-2][1], covered)
    ref = spans_to_flags(ref_spans, len(bolmo))
    diffs = []
    agree = 0
    for p in range(first, end_eval):
        if bolmo[p] == ref[p]:
            agree += 1
        else:
            diffs.append((p, int(bolmo[p]), ref[p]))
    tokset = set(ref_spans)
    patches = exact = 0
    start = 0
    for p in range(end_eval):
        if bolmo[p]:
            if p >= first:
                patches += 1
                exact += (start, p + 1) in tokset
            start = p + 1
    return {"positions": max(end_eval - first, 0), "agree": agree, "patches": patches, "patches_exact": exact,
            "diffs": diffs}


def classify_diff(data: bytes, p: int, bolmo_flag: int) -> str:
    """Kind of a flag disagreement at byte position p: direction + the characters around the cut."""
    direction = "extra" if bolmo_flag else "missing"  # Bolmo cuts where dolma2 does not / does not cut where dolma2 does
    a = data[p]
    b = data[p + 1] if p + 1 < len(data) else 0x20
    if a >= 0x80 or b >= 0x80:
        kind = "multibyte"
    elif chr(a).isdigit() or chr(b).isdigit():
        kind = "digit"
    elif chr(a).isspace() or chr(b).isspace():
        kind = "whitespace"
    elif chr(a).isalnum() and chr(b).isalnum():
        kind = "inside-word"
    else:
        kind = "punctuation"
    return f"{direction}/{kind}"
