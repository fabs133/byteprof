"""Byte n-gram model over a fixed corpus, greedy continuation (backs off to shorter contexts).

For every order n the corpus is reduced to "context hash -> most frequent next atom (byte, flag)"
with its count (capped at 255). The tables are direct-mapped (2**bits slots, 8-bit check, a higher
count wins a collision), so a lookup is a few array reads. A context's hash is a prefix sum over the
last bytes, so one pass over the last `max(orders)` bytes yields the hash of every order.

The byte is the argmax over bytes; its flag is the majority flag among occurrences of that
(context, byte) pair, so flags do not split the byte counts.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence

import numpy as np

M64 = (1 << 64) - 1
P = 1099511628211
MUL = 0x9E3779B97F4A7C15
_P64 = np.uint64(P)
_MUL64 = np.uint64(MUL)
KEY_BITS = 52  # context key = hash >> 12; table index = its top `bits` bits, check = the next 8 bits


def _pows(n: int) -> list[int]:
    return [pow(P, t, 1 << 64) for t in range(n + 1)]


class NGramModel:
    def __init__(self, orders: Sequence[int], bits: int):
        self.orders = tuple(sorted(orders))
        self.bits = bits
        self.atom: dict[int, array] = {}  # order -> array('H'): atom + 1, 0 = empty
        self.chk: dict[int, bytes] = {}
        self.cnt: dict[int, bytes] = {}
        self._pw = _pows(max(self.orders))
        self.shift = KEY_BITS - bits
        self.n_entries: dict[int, int] = {}

    # -- build --------------------------------------------------------------------------
    @classmethod
    def build(cls, data: bytes, flags: np.ndarray, orders: Sequence[int] = (1, 2, 3, 4, 6, 8),
              bits: int = 24, shards: int = 4) -> "NGramModel":
        """`shards` splits every order by the top bits of the context key, which bounds the peak RAM."""
        m = cls(orders, bits)
        d = np.frombuffer(data, np.uint8)
        fl = np.asarray(flags, np.uint8)
        N = len(d)
        pw = [np.uint64(p) for p in m._pw]
        size = 1 << bits
        sbits = max(shards - 1, 0).bit_length()
        for n in m.orders:
            h = np.zeros(N - n, np.uint64)
            with np.errstate(over="ignore"):
                for t in range(1, n + 1):
                    h += (d[n - t:N - t].astype(np.uint64) + np.uint64(1)) * pw[t]
                h *= _MUL64
            h >>= np.uint64(64 - KEY_BITS)  # 52-bit context key, in place
            key = h.view(np.int64)
            t_atom = np.zeros(size, np.uint16)
            t_chk = np.zeros(size, np.uint8)
            t_cnt = np.zeros(size, np.uint8)
            n_entries = 0
            for s in range(1 << sbits):
                sel = np.flatnonzero((key >> (KEY_BITS - sbits)) == s) if sbits else np.arange(len(key))
                if len(sel) == 0:
                    continue
                kb = (key[sel] << 8) | d[n:][sel].astype(np.int64)
                nf = fl[n:][sel]
                del sel
                uk, inv, cnt = np.unique(kb, return_inverse=True, return_counts=True)
                del kb
                fsum = np.bincount(inv, weights=nf, minlength=len(uk))
                del inv, nf
                fbest = (2 * fsum > cnt).astype(np.int64)
                del fsum
                group = uk >> 8
                packed = (cnt.astype(np.int64) << 9) | (fbest << 8) | (uk & 255)
                del uk, cnt, fbest
                starts = np.flatnonzero(np.concatenate([[True], group[1:] != group[:-1]]))
                best = np.maximum.reduceat(packed, starts)
                gk = group[starts]
                del group, packed
                atom = best & 511
                count = np.minimum(best >> 9, 255)
                tidx = (gk >> m.shift).astype(np.int64)
                chk = ((gk >> (m.shift - 8)) & 255).astype(np.uint8)
                order = np.argsort(count, kind="stable")  # ascending: the highest count is written last
                t_atom[tidx[order]] = (atom[order] + 1).astype(np.uint16)
                t_chk[tidx[order]] = chk[order]
                t_cnt[tidx[order]] = count[order].astype(np.uint8)
                n_entries += len(gk)
                del best, gk, atom, count, tidx, chk, order
            del h, key
            m.n_entries[n] = n_entries
            a = array("H")
            a.frombytes(t_atom.tobytes())
            m.atom[n], m.chk[n], m.cnt[n] = a, t_chk.tobytes(), t_cnt.tobytes()
            del t_atom, t_chk, t_cnt
        return m

    # -- query ----------------------------------------------------------------------------
    def next_atom(self, tail: Sequence[int], min_count: int = 1, max_order: int | None = None) -> int:
        """Most likely next atom given the last bytes in `tail`, or -1."""
        pw, nt = self._pw, len(tail)
        hs = [0]
        h = 0
        top = min(self.orders[-1] if max_order is None else max_order, nt)
        for t in range(1, top + 1):
            h = (h + (tail[-t] + 1) * pw[t]) & M64
            hs.append(h)
        for n in reversed(self.orders):
            if n > top:
                continue
            key = ((hs[n] * MUL) & M64) >> 12
            ti = key >> self.shift
            if self.chk[n][ti] == ((key >> (self.shift - 8)) & 255) and self.cnt[n][ti] >= min_count:
                a = self.atom[n][ti]
                if a:
                    return a - 1
        return -1


class NGramDrafter:
    """Greedy continuation of the context from an `NGramModel`."""

    name = "ngram"

    def __init__(self, model: NGramModel, min_count: int = 1, max_order: int | None = None):
        self.model, self.min_count, self.max_order = model, min_count, max_order
        self._keep = model.orders[-1]
        self.tail: list[int] = []

    def reset(self, ctx_atoms: Sequence[int], truth: Sequence[int] | None = None) -> None:
        self.tail = [a & 255 for a in ctx_atoms[-self._keep:]]

    def push(self, atoms: Sequence[int]) -> None:
        self.tail.extend(a & 255 for a in atoms)
        if len(self.tail) > 4 * self._keep:
            del self.tail[:-self._keep]

    def propose(self, k: int) -> list[int]:
        tail = self.tail[-self._keep:]
        out: list[int] = []
        nxt = self.model.next_atom
        for _ in range(k):
            a = nxt(tail, self.min_count, self.max_order)
            if a < 0:
                break
            out.append(a)
            tail.append(a & 255)
            if len(tail) > self._keep:
                del tail[0]
        return out
