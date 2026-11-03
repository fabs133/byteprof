"""Drafter diagnostics (pure functions, no model): why a drafter accepts more for OLMo 2 than for Bolmo.

Two round rules replay the same recorded drafter continuations:

* byte rule (Bolmo, `sim.simulate` semantics): a draft of at most K bytes; byte and boundary flag must match; one
  byte (the model's own prediction) is always emitted in addition to the accepted prefix;
* token rule (OLMo 2, `tokens.simulate_tokens` semantics): a draft of at most K_tok dolma2 tokens; every token must
  match; one *token* (the model's own prediction, about 3.7 bytes) is always emitted in addition.

`replay_byte_rounds` / `replay_token_rounds` return one record per round with the reason the round ended.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from . import byte_of, flag_of, make_atom
from .sim import accepted_prefix

BYTE_ENDS = ("wrong_byte", "wrong_flag", "patch_end", "exhausted", "cap", "patch_cap", "end_of_generation")
TOKEN_ENDS = ("wrong_token", "exhausted", "cap", "end_of_generation")
POSITIONS = ("word_start", "inside_word", "whitespace_punct", "multibyte")


@dataclass
class RoundRec:
    i: int  # index of the last emitted unit before the round (an atom for the byte rule, a token for the token rule)
    draft_len: int  # units offered
    L: int  # accepted draft units
    end: str  # one of BYTE_ENDS / TOKEN_ENDS
    emitted: int  # units emitted (1 + L, truncated at the end of generation)
    nbytes: int  # bytes emitted
    miss_i: int | None = None  # index into the truth of the first wrong unit (wrong_* ends only)
    miss_draft: bytes = b""  # the draft's bytes from the first wrong unit on (byte rule), at most 16
    miss_truth: bytes = b""  # the truth's bytes from the first wrong unit on, same length
    extra: dict = field(default_factory=dict)


def classify_end(draft: Sequence[int], preds: Sequence[int], k: int, multi_patch: bool, *, room: int | None = None,
                 patch_truncated: bool = False, more_available: bool = False) -> tuple[int, str]:
    """(L, why the round ends) for the byte rule. `preds[j-1]` = p_j. `room` = units still to be generated;
    `patch_truncated` = the draft was cut by the patch limit; `more_available` = the drafter had more atoms than
    were offered (the draft was cut to K, or by the retokenization seam)."""
    L = accepted_prefix(draft, preds, k, multi_patch)
    n = min(k, len(draft), len(preds))
    if room is not None and 1 + L > room:
        return L, "end_of_generation"
    if L < n:
        d, p = draft[L], preds[L]
        if byte_of(d) != byte_of(p):
            return L, "wrong_byte"
        return L, "wrong_flag" if multi_patch else "patch_end"
    if patch_truncated:
        return L, "patch_cap"
    return L, "cap" if (more_available or len(draft) >= k) else "exhausted"


def limit_patches(draft: list[int], n_patches0: int, max_patches: int) -> tuple[list[int], bool]:
    """Cut a draft before the flagged atom that would be patch number max_patches + 1 (as `sim.simulate` does)."""
    n_p = n_patches0
    for j, dj in enumerate(draft):
        if flag_of(dj):
            if n_p + 1 > max_patches:
                return draft[:j], True
            n_p += 1
    return draft, False


def replay_byte_rounds(truth: Sequence[int], drafter, k: int, *, ctx_atoms: Sequence[int] = (), max_new: int | None = None,
                       multi_patch: bool = True, max_patches: int | None = None, miss_window: int = 16,
                       align: set[int] | None = None) -> list[RoundRec]:
    """The rounds of `sim.simulate` (same rule, no `ks`), one record each. `drafter.propose(1 << 20)` is used to tell
    a drafter that has nothing more to offer from a draft cut to K.

    `align` (a what-if, not a Bolmo rule): a set of truth indices that are token ends. When given, a round that ends
    off a token end also emits the rest of the token, as the bonus *token* of OLMo's rule does, so every round starts
    at a token boundary. It separates the effect of the bonus unit from the effect of matching bytes instead of
    tokens."""
    n_avail = len(truth) - 1
    n = n_avail if max_new is None else min(max_new, n_avail)
    drafter.reset(list(ctx_atoms) + [truth[0]], truth)
    recs: list[RoundRec] = []
    emitted = 0
    i = 0
    while emitted < n:
        full = drafter.propose(1 << 20) if k > 0 else []
        draft = list(drafter.propose(k))[:k] if k > 0 else []
        truncated = False
        if multi_patch and max_patches is not None:
            draft, truncated = limit_patches(draft, flag_of(truth[i]), max_patches)
        preds = truth[i + 1: i + 2 + k]
        L, end = classify_end(draft, preds, k, multi_patch, room=n - emitted, patch_truncated=truncated,
                              more_available=len(full) > len(draft))
        take = min(1 + L, n - emitted)
        if align is not None and (i + take) not in align:
            nxt = min((e for e in align if e > i + take), default=n)
            take = min(nxt - i, n - emitted)
        rec = RoundRec(i, len(draft), L, end, take, take)
        if end in ("wrong_byte", "wrong_flag", "patch_end") and L < len(preds):
            rec.miss_i = i + 1 + L
            rec.miss_draft = bytes(byte_of(a) for a in draft[L:L + miss_window])
            rec.miss_truth = bytes(byte_of(a) for a in truth[i + 1 + L: i + 1 + L + miss_window])
        recs.append(rec)
        drafter.push(truth[i + 1: i + 1 + take])
        emitted += take
        i += take
    return recs


def replay_token_rounds(truth: Sequence[int], drafter, k_tok: int, tok_nbytes: Callable[[int], int], *,
                        max_new: int | None = None) -> list[RoundRec]:
    """The rounds of `tokens.simulate_tokens` with a reason per round. `truth[0]` is already emitted;
    `tok_nbytes(token)` = byte length of a token. `drafter` must be reset (`set_text`) by the caller."""
    n_avail = len(truth) - 1
    n = n_avail if max_new is None else min(max_new, n_avail)
    drafter.reset([truth[0]], truth)
    recs: list[RoundRec] = []
    emitted = 0
    i = 0
    while emitted < n:
        full = drafter.propose(1 << 20) if k_tok > 0 else []
        draft = list(drafter.propose(k_tok))[:k_tok] if k_tok > 0 else []
        L = 0
        while L < len(draft) and i + 1 + L < len(truth) and draft[L] == truth[i + 1 + L]:
            L += 1
        take = min(1 + L, n - emitted)
        if 1 + L > n - emitted:
            end = "end_of_generation"
        elif L < len(draft) and i + 1 + L < len(truth):
            end = "wrong_token"
        else:
            end = "cap" if (len(full) > len(draft) or len(draft) >= k_tok) else "exhausted"
        nb = sum(tok_nbytes(t) for t in truth[i + 1: i + 1 + take])
        rec = RoundRec(i, len(draft), L, end, take, nb)
        rec.extra["acc_bytes"] = sum(tok_nbytes(t) for t in truth[i + 1: i + 1 + min(L, take)])  # without the bonus token
        if end == "wrong_token":
            rec.miss_i = i + 1 + L
        recs.append(rec)
        drafter.push(truth[i + 1: i + 1 + take])
        emitted += take
        i += take
    return recs


def end_shares(recs: Sequence[RoundRec], kinds: Sequence[str]) -> dict[str, float]:
    n = max(len(recs), 1)
    return {k: sum(r.end == k for r in recs) / n for k in kinds}


def per_round(recs: Sequence[RoundRec]) -> dict[str, float]:
    n = max(len(recs), 1)
    return {"rounds": len(recs), "units": sum(r.emitted for r in recs) / n, "bytes": sum(r.nbytes for r in recs) / n,
            "accepted_units": sum(r.L for r in recs) / n}


# --- text helpers ------------------------------------------------------------------------------------------

def atoms_with_flags(data: bytes, flags: Sequence[int]) -> list[int]:
    return [make_atom(b, f) for b, f in zip(data, flags)]


def retokenize(dol, data: bytes) -> tuple[list[int], list[tuple[int, int]], int]:
    """(token ids, byte spans, bytes covered) of `data` (an incomplete trailing UTF-8 character is left out)."""
    try:
        data.decode("utf-8")
        used = len(data)
    except UnicodeDecodeError as e:
        used = e.start
    ids = dol.encode_ids(data[:used].decode("utf-8"))
    spans: list[tuple[int, int]] = []
    pos = 0
    for t in ids:
        nb = len(dol.id_bytes(t))
        spans.append((pos, pos + nb))
        pos += nb
    assert pos == used, (pos, used)
    return ids, spans, used


def token_truth(dol, data: bytes, prompt_len: int) -> dict:
    """The token reference for the token rule on a text: the tokens of `data` that start at or after the prompt
    end, without the last (possibly incomplete) one. truth[0] = the first of them (already emitted)."""
    ids, spans, _ = retokenize(dol, data)
    sel = [j for j in range(len(ids) - 1) if spans[j][0] >= prompt_len]
    return {"ids": [ids[j] for j in sel], "spans": [spans[j] for j in sel]}


def classify_position(data: bytes, p: int) -> str:
    """Where byte `p` of `data` sits: word_start, inside_word, whitespace_punct or multibyte (any byte >= 0x80)."""
    b = data[p]
    if b >= 0x80:
        return "multibyte"
    c = chr(b)
    if c.isalnum() or c == "_":
        prev = data[p - 1] if p > 0 else 0x20
        return "inside_word" if (prev >= 0x80 or chr(prev).isalnum() or chr(prev) == "_") else "word_start"
    return "whitespace_punct"


def ws_near_miss(draft_rest: bytes, truth_rest: bytes, min_cmp: int = 4, window: int = 12) -> bool:
    """The first miss is only a whitespace difference: with ASCII whitespace removed, the draft and the truth agree
    over at least `min_cmp` bytes (up to `window`) from the miss on, while the raw bytes differ."""
    if draft_rest[:window] == truth_rest[:window]:
        return False
    ws = bytes(b" \t\r\n\x0b\x0c")
    d = draft_rest.translate(None, ws)[:window]
    t = truth_rest.translate(None, ws)[:window]
    m = min(len(d), len(t))
    return m >= min_cmp and d[:m] == t[:m]


def next_token_agreement(draft_token: bytes, actual_token: bytes) -> tuple[bool, bool]:
    """(agree up to the shorter length, bytes identical). Empty tokens never agree."""
    if not draft_token or not actual_token:
        return False, False
    m = min(len(draft_token), len(actual_token))
    return draft_token[:m] == actual_token[:m], draft_token == actual_token


def first_token_bytes(cont, healed_back: bytes = b"") -> bytes:
    """The first token of a recorded continuation (a `lmdraft.Cont`), after the held-back fragment."""
    if cont is None or not cont.tok_ends:
        return b""
    start = len(healed_back)
    for e in cont.tok_ends:
        if e > start:
            return cont.cont[start:e]
    return b""


def predicted_accept_tokens(p: float) -> float:
    """Expected accepted tokens per round when every token matches independently with probability p."""
    return p / (1 - p) if p < 1 else float("inf")


def bits_per_byte(nll_nats: Sequence[float], nbytes: Sequence[int]) -> float:
    import math

    tot = sum(nbytes)
    return sum(nll_nats) / math.log(2) / tot if tot else float("nan")


def iter_token_boundaries(spans: Sequence[tuple[int, int]]) -> Iterator[int]:
    for _, e in spans:
        yield e


_SPACE_FRAGMENT = re.compile(r"[ \t]?\w*\Z")  # \Z, not $: "$" also matches before a trailing newline


def context_back_space(data: bytes) -> int:
    """Token healing as a BPE model sees words: hold back the trailing word fragment *with its leading space*
    (or a lone trailing space), plus an incomplete UTF-8 character. `lmdraft.context_back(healed=True)` holds back
    the fragment only, so the drafter's context then ends in a bare space, which BPE models rarely continue."""
    from .lmdraft import utf8_valid_prefix

    valid = utf8_valid_prefix(data)
    back = len(data) - valid
    m = _SPACE_FRAGMENT.search(data[:valid].decode("utf-8"))
    if m:
        back += len(m.group(0).encode("utf-8"))
    return back
