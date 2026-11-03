"""SubwordAutomaton vs. the reference semantics of BolmoTokenizer.expand_byte_ids.

The algorithm tests use a synthetic vocabulary and a clone of the reference trie walk (no model
download). `test_real_tokenizer_equivalence` uses the real Bolmo tokenizer and is skipped when
it is not in the local HF cache.
"""

import json
import random
from pathlib import Path

import pytest

from byteprof.optim.subword_automaton import NoMatchError, SubwordAutomaton

OFFSET = 4
BOUNDARY = OFFSET + 256
BPE_END = 3
TOKEN_ID_KEY = -1


class ReferenceTrie:
    """Line-by-line clone of BolmoTokenizer's trie construction and expand_byte_ids(n_last=1)."""

    def __init__(self, byte_sequences):
        self.trie = {}
        for token_id, seq in byte_sequences.items():
            d = self.trie
            for b in seq[::-1]:
                d = d.setdefault(b, {})
            d[TOKEN_ID_KEY] = token_id

    def last(self, history):
        i0 = len(history) - 1
        cur, exp = self.trie, None
        for i in range(i0, -1, -1):
            b = history[i]
            if b == BPE_END:
                continue
            if b >= BOUNDARY:
                b -= BOUNDARY
            try:
                cur = cur[b]
                if TOKEN_ID_KEY in cur:
                    exp = cur[TOKEN_ID_KEY]
            except KeyError:
                assert exp is not None
                break
        return exp


def brute_force(byte_sequences, history):
    """Longest entry that is a suffix of the stripped history (last inserted wins on ties)."""
    h = [b - BOUNDARY if b >= BOUNDARY else b for b in history if b != BPE_END]
    best, best_len = None, -1
    for token_id, seq in byte_sequences.items():
        n = len(seq)
        if n <= len(h) and list(seq) == h[len(h) - n:] and n >= best_len:
            best, best_len = token_id, n
    return best


def synthetic_vocab(rng, n_tokens=300, alphabet=range(OFFSET, OFFSET + 6), max_len=6):
    seqs = {}
    tid = 0
    for b in range(OFFSET, OFFSET + 6):  # every single symbol is an entry (byte-level base vocab)
        seqs[tid] = [b]
        tid += 1
    seqs[tid] = [1]  # a special entry
    tid += 1
    for _ in range(n_tokens):
        seqs[tid] = [rng.choice(list(alphabet)) for _ in range(rng.randint(2, max_len))]
        tid += 1
    return seqs


def random_history(rng, n, alphabet=range(OFFSET, OFFSET + 6)):
    out = []
    for _ in range(n):
        r = rng.random()
        b = rng.choice(list(alphabet))
        if r < 0.15:
            b += BOUNDARY  # fused boundary flag: plain id + offset + 256
        out.append(b)
    return out


def test_matches_reference_and_brute_force_on_random_strings():
    rng = random.Random(0)
    for trial in range(20):
        vocab = synthetic_vocab(rng)
        ref = ReferenceTrie(vocab)
        auto = SubwordAutomaton(vocab, offset=OFFSET, bpe_token_end_id=BPE_END)
        hist = random_history(rng, 300)
        state = 0
        for i, b in enumerate(hist):
            state, sid = auto.step(state, b)
            assert sid == ref.last(hist[: i + 1]) == brute_force(vocab, hist[: i + 1]), (trial, i)


def test_duplicate_sequences_last_wins():
    vocab = {0: [OFFSET], 1: [OFFSET + 1], 2: [OFFSET + 1]}  # ids 1 and 2 share a sequence
    auto = SubwordAutomaton(vocab, offset=OFFSET, bpe_token_end_id=BPE_END)
    assert auto.step(0, OFFSET + 1)[1] == 2 == ReferenceTrie(vocab).last([OFFSET + 1])


def test_bpe_token_end_is_skipped():
    vocab = synthetic_vocab(random.Random(1))
    ref = ReferenceTrie(vocab)
    auto = SubwordAutomaton(vocab, offset=OFFSET, bpe_token_end_id=BPE_END)
    hist = [OFFSET, OFFSET + 1, BPE_END, OFFSET + 2, BPE_END, BPE_END, OFFSET + 1]
    state = 0
    for i, b in enumerate(hist):
        state, sid = auto.step(state, b)
        assert sid == ref.last(hist[: i + 1])


def test_multibyte_utf8_entries():
    texts = ["é", "日", "日本", "本語", "😀", "a", "b", " ", "ab", "日本語"]
    singles = {bytes([x]) for x in range(256)}
    entries = [t.encode() for t in texts] + sorted(singles)
    vocab = {i: [OFFSET + b for b in e] for i, e in enumerate(entries)}
    ref = ReferenceTrie(vocab)
    auto = SubwordAutomaton(vocab, offset=OFFSET, bpe_token_end_id=BPE_END)
    data = "ab日本語😀é a日本".encode()
    hist = [OFFSET + b for b in data]
    state = 0
    got = []
    for i, b in enumerate(hist):
        state, sid = auto.step(state, b)
        got.append(sid)
        assert sid == ref.last(hist[: i + 1])
    # the full three-character entry is found once its last byte arrives
    full = entries.index("日本語".encode())
    assert full in got


def test_no_match_raises():
    vocab = {0: [OFFSET], 1: [OFFSET, OFFSET + 1]}  # byte OFFSET+2 is not an entry
    auto = SubwordAutomaton(vocab, offset=OFFSET, bpe_token_end_id=BPE_END)
    state, sid = auto.step(0, OFFSET)
    assert sid == 0
    with pytest.raises(NoMatchError):
        auto.step(state, OFFSET + 2)
    with pytest.raises(AssertionError):  # the reference asserts in the same situation
        ReferenceTrie(vocab).last([OFFSET, OFFSET + 2])


def test_state_is_independent_of_history_length():
    """O(1) per step: the state after the same suffix is the same however long the prefix."""
    rng = random.Random(2)
    vocab = synthetic_vocab(rng)
    auto = SubwordAutomaton(vocab, offset=OFFSET, bpe_token_end_id=BPE_END)
    tail = random_history(rng, 8)
    s1, _ = auto.run(tail)
    s2, _ = auto.run(random_history(rng, 500) + tail)
    # a longer suffix-prefix may survive from the long prefix only if it is part of an entry
    assert auto._best[s1] == auto._best[s2]


# ---- real tokenizer -----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def _real_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "allenai/Bolmo-1B", revision="452690a8cfab1dd1ec27b74f2667632f17a21b46",
            trust_remote_code=True, local_files_only=True)
    except Exception as exc:  # not downloaded
        pytest.skip(f"Bolmo tokenizer not available locally: {exc}")


@pytest.mark.real_tokenizer
def test_real_tokenizer_equivalence():
    tok = _real_tokenizer()
    auto = SubwordAutomaton.from_tokenizer(tok)
    off = tok.offset
    corpora: list[tuple[str, list[int]]] = []

    prompts = [json.loads(l)["text"] for l in (ROOT / "prompts/default.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, text in enumerate(prompts):
        corpora.append((f"prompt{i}", tok(text)["input_ids"]))  # includes BOS and real ids

    gen = ROOT / "tests/data/bolmo_generated.json"
    if gen.is_file():  # texts generated by Bolmo in earlier runs (written by tools/dump_bolmo_texts.py)
        for k, text in enumerate(json.loads(gen.read_text(encoding="utf-8"))):
            corpora.append((f"generated{k}", tok(text)["input_ids"]))

    rng = random.Random(1234)
    corpora.append(("random-bytes", [1] + [off + rng.randrange(256) + (off + 256 if rng.random() < 0.3 else 0) for _ in range(12000)]))
    utf8 = []
    while len(utf8) < 12000:
        utf8 += [off + b for b in chr(rng.choice([rng.randrange(0x20, 0x7F), rng.randrange(0xA0, 0x800),
                                                  rng.randrange(0x800, 0xD800), rng.randrange(0x10000, 0x1F700)])).encode()]
    corpora.append(("random-utf8", [1] + [b + (off + 256 if rng.random() < 0.3 else 0) for b in utf8]))

    total = 0
    for name, ids in corpora:
        state = 0
        expected = tok.expand_byte_ids(ids)  # reference for every position
        for i, b in enumerate(ids):
            state, sid = auto.step(state, b)
            assert sid == expected[i], (name, i)
        total += len(ids)
    assert total >= 24000
