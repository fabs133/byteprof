"""The same drafter family on token ids, for OLMo 2 (Stage 3): context lookup on tokens, a token n-gram built from
the same corpus (segmented with OLMo's tokenizer, see corpus.segment_with_tokenizer), and a token round simulator.

Interfaces match `sim.Drafter` (reset / propose / push), with token ids as "atoms"."""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

M64 = (1 << 64) - 1
P = 1099511628211
MUL = 0x9E3779B97F4A7C15
KEY_BITS = 44
SYM_BITS = 17  # OLMo 2's vocabulary (100,278) fits in 17 bits


class TokenContextDrafter:
    """Prompt-lookup on token ids: the continuation of the longest earlier occurrence of the context's suffix."""

    name = "context"

    def __init__(self, min_len: int = 2, max_candidates: int = 8, max_match: int = 32):
        self.min_len, self.max_cand, self.max_match = min_len, max_candidates, max_match
        self.buf: list[int] = []
        self.index: dict[tuple[int, ...], list[int]] = {}
        self.last_match = 0

    def reset(self, ctx: Sequence[int], truth: Sequence[int] | None = None) -> None:
        self.buf, self.index = [], {}
        self.push(ctx)

    def push(self, toks: Sequence[int]) -> None:
        buf, index, m = self.buf, self.index, self.min_len
        for t in toks:
            buf.append(t)
            n = len(buf)
            if n >= m:
                index.setdefault(tuple(buf[n - m:n]), []).append(n - 1)

    def propose(self, k: int) -> list[int]:
        buf = self.buf
        n = len(buf)
        self.last_match = 0
        if n < self.min_len or k <= 0:
            return []
        cands = self.index.get(tuple(buf[n - self.min_len:n]))
        if not cands:
            return []
        best_e, best_l, seen = -1, 0, 0
        for e in reversed(cands):
            if e == n - 1:
                continue
            seen += 1
            if seen > self.max_cand:
                break
            l, cap = 0, min(self.max_match, e + 1)
            while l < cap and buf[e - l] == buf[n - 1 - l]:
                l += 1
            if l > best_l:
                best_e, best_l = e, l
        if best_e < 0:
            return []
        self.last_match = best_l
        out: list[int] = []
        for t in range(k):
            idx = best_e + 1 + t
            out.append(buf[idx] if idx < n else out[idx - n])
        return out


class TokenNGramModel:
    """Direct-mapped hash tables per order: context hash -> most frequent next token (+ count, 8-bit check)."""

    def __init__(self, orders: Sequence[int], bits: int):
        self.orders, self.bits = tuple(sorted(orders)), bits
        self.shift = KEY_BITS - bits
        self.sym: dict[int, array] = {}
        self.chk: dict[int, bytes] = {}
        self.cnt: dict[int, bytes] = {}
        self._pw = [pow(P, t, 1 << 64) for t in range(max(self.orders) + 1)]
        self.n_entries: dict[int, int] = {}

    @classmethod
    def build(cls, tokens: np.ndarray, orders: Sequence[int] = (1, 2, 3, 4), bits: int = 22) -> "TokenNGramModel":
        m = cls(orders, bits)
        d = np.asarray(tokens, np.int64)
        N = len(d)
        pw = [np.uint64(p) for p in m._pw]
        size = 1 << bits
        for n in m.orders:
            h = np.zeros(N - n, np.uint64)
            with np.errstate(over="ignore"):
                for t in range(1, n + 1):
                    h += (d[n - t:N - t].astype(np.uint64) + np.uint64(1)) * pw[t]
                h *= np.uint64(MUL)
            h >>= np.uint64(64 - KEY_BITS)
            key = h.view(np.int64)
            kb = (key << SYM_BITS) | d[n:]
            uk, cnt = np.unique(kb, return_counts=True)
            del kb, h, key
            group = uk >> SYM_BITS
            packed = (cnt.astype(np.int64) << SYM_BITS) | (uk & ((1 << SYM_BITS) - 1))
            starts = np.flatnonzero(np.concatenate([[True], group[1:] != group[:-1]]))
            best = np.maximum.reduceat(packed, starts)
            gk = group[starts]
            sym = best & ((1 << SYM_BITS) - 1)
            count = np.minimum(best >> SYM_BITS, 255)
            tidx = (gk >> m.shift).astype(np.int64)
            chk = ((gk >> (m.shift - 8)) & 255).astype(np.uint8)
            order = np.argsort(count, kind="stable")
            t_sym = np.zeros(size, np.uint32)
            t_chk = np.zeros(size, np.uint8)
            t_cnt = np.zeros(size, np.uint8)
            t_sym[tidx[order]] = (sym[order] + 1).astype(np.uint32)
            t_chk[tidx[order]] = chk[order]
            t_cnt[tidx[order]] = count[order].astype(np.uint8)
            m.n_entries[n] = int(len(gk))
            a = array("I")
            a.frombytes(t_sym.tobytes())
            m.sym[n], m.chk[n], m.cnt[n] = a, t_chk.tobytes(), t_cnt.tobytes()
        return m

    def next_token(self, tail: Sequence[int], min_count: int = 1, max_order: int | None = None) -> int:
        pw, nt = self._pw, len(tail)
        top = min(self.orders[-1] if max_order is None else max_order, nt)
        hs = [0]
        h = 0
        for t in range(1, top + 1):
            h = (h + (tail[-t] + 1) * pw[t]) & M64
            hs.append(h)
        for n in reversed(self.orders):
            if n > top:
                continue
            key = ((hs[n] * MUL) & M64) >> (64 - KEY_BITS)
            ti = key >> self.shift
            if self.chk[n][ti] == ((key >> (self.shift - 8)) & 255) and self.cnt[n][ti] >= min_count:
                s = self.sym[n][ti]
                if s:
                    return s - 1
        return -1


class TokenNGramDrafter:
    name = "ngram"

    def __init__(self, model: TokenNGramModel, min_count: int = 1, max_order: int | None = None):
        self.model, self.min_count, self.max_order = model, min_count, max_order
        self._keep = model.orders[-1]
        self.tail: list[int] = []

    def reset(self, ctx: Sequence[int], truth: Sequence[int] | None = None) -> None:
        self.tail = list(ctx[-self._keep:])

    def push(self, toks: Sequence[int]) -> None:
        self.tail.extend(toks)
        if len(self.tail) > 4 * self._keep:
            del self.tail[:-self._keep]

    def propose(self, k: int) -> list[int]:
        tail = self.tail[-self._keep:]
        out: list[int] = []
        for _ in range(k):
            t = self.model.next_token(tail, self.min_count, self.max_order)
            if t < 0:
                break
            out.append(t)
            tail.append(t)
            if len(tail) > self._keep:
                del tail[0]
        return out


@dataclass
class TokenSim:
    k: int
    emitted: int = 0
    rounds: int = 0
    l_hist: dict[int, int] = field(default_factory=dict)
    by_k: dict[int, int] = field(default_factory=dict)

    @property
    def tokens_per_round(self) -> float:
        return self.emitted / self.rounds if self.rounds else 0.0


def simulate_tokens(truth: Sequence[int], drafter, k: int, *, ctx: Sequence[int] = (), max_new: int | None = None,
                    ks: Sequence[int] | None = None) -> TokenSim:
    """Round rule for tokens: a0 = truth[0] is already emitted; a draft d_1..d_K is accepted up to the first
    token that differs from the prediction; 1 + L tokens are emitted per round (no boundaries in a token model)."""
    n_avail = len(truth) - 1
    n = n_avail if max_new is None else min(max_new, n_avail)
    res = TokenSim(k)
    if drafter is not None:
        drafter.reset(list(ctx) + [truth[0]], truth)
    i = 0
    while res.emitted < n:
        draft = list(drafter.propose(k))[:k] if (drafter is not None and k > 0) else []
        if ks is not None:
            draft = draft[:max([kv for kv in ks if kv <= len(draft)], default=0)]
        L = 0
        while L < len(draft) and i + 1 + L < len(truth) and draft[L] == truth[i + 1 + L]:
            L += 1
        take = min(1 + L, n - res.emitted)
        res.rounds += 1
        res.by_k[len(draft)] = res.by_k.get(len(draft), 0) + 1
        res.l_hist[L] = res.l_hist.get(L, 0) + 1
        if drafter is not None:
            drafter.push(truth[i + 1: i + 1 + take])
        res.emitted += take
        i += take
    return res
