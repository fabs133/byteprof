"""Incremental subword-suffix lookup for Bolmo (pure Python, no torch).

`BolmoTokenizer.expand_byte_ids(history, n_last=1)` returns, for the last byte of `history`,
the id of the longest vocabulary entry that is a suffix of the history. It finds it by walking
the whole history backwards through a trie of reversed entries, and callers pass the full
history every step, so decoding n bytes costs O(n^2).

Semantics reproduced here (read from tokenization_bolmo.py):
- an entry is a sequence of ids in [0, offset + 256): special ids, then byte + offset;
- history ids >= offset + 256 are bytes with a fused boundary flag: the flag is dropped;
- history ids equal to `bpe_token_end_id` are skipped entirely;
- if two vocabulary entries have the same id sequence, the one inserted later wins;
- no entry matching raises (the reference asserts).

An Aho-Corasick automaton over the entries gives the same answer incrementally: its state is
the longest history suffix that is a prefix of an entry, and the answer is the longest entry
among that state's suffixes (`best[state]`, folded in while building). One `step()` is O(1)
amortized.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

_SHIFT = 10  # state * 1024 + symbol; symbols are < 1024


class NoMatchError(LookupError):
    """No vocabulary entry is a suffix of the history (the reference asserts in this case)."""


class SubwordAutomaton:
    ROOT = 0

    def __init__(self, sequences: Mapping[int, Sequence[int]] | Iterable[tuple[int, Sequence[int]]],
                 *, offset: int, bpe_token_end_id: int):
        items = sequences.items() if isinstance(sequences, Mapping) else sequences
        self.offset = offset
        self.boundary_offset = offset + 256
        self.bpe_token_end_id = bpe_token_end_id

        # later insertion wins for identical sequences, as in the reference trie
        by_seq: dict[tuple[int, ...], int] = {}
        for token_id, seq in items:
            by_seq[tuple(seq)] = token_id

        edges: dict[int, int] = {}
        parent = [0]
        sym = [0]
        depth = [0]
        terminal = [-1]
        for seq, token_id in by_seq.items():
            node = 0
            for c in seq:
                key = (node << _SHIFT) | c
                nxt = edges.get(key)
                if nxt is None:
                    nxt = len(parent)
                    edges[key] = nxt
                    parent.append(node)
                    sym.append(c)
                    depth.append(depth[node] + 1)
                    terminal.append(-1)
                node = nxt
            terminal[node] = token_id

        n = len(parent)
        order = sorted(range(1, n), key=depth.__getitem__)  # parents before children
        fail = [0] * n
        best = [-1] * n
        best[0] = terminal[0]
        for v in order:
            p = parent[v]
            if p == 0:
                fail[v] = 0
            else:
                c = sym[v]
                f = fail[p]
                while True:
                    nxt = edges.get((f << _SHIFT) | c)
                    if nxt is not None:
                        fail[v] = nxt
                        break
                    if f == 0:
                        fail[v] = 0
                        break
                    f = fail[f]
            best[v] = terminal[v] if terminal[v] >= 0 else best[fail[v]]
        self._edges, self._fail, self._best = edges, fail, best
        self._memo: dict[int, int] = {}
        self.n_states, self.n_entries = n, len(by_seq)

    @classmethod
    def from_tokenizer(cls, tok) -> "SubwordAutomaton":
        """Build from a BolmoTokenizer (uses its `byte_sequences`, `offset`, `bpe_token_end_id`)."""
        return cls(tok.byte_sequences, offset=tok.offset, bpe_token_end_id=tok.bpe_token_end_id)

    def step(self, state: int, byte: int) -> tuple[int, int]:
        """Feed one history id. Returns (new_state, subword_id of the longest entry ending here)."""
        if byte == self.bpe_token_end_id:
            return state, self._answer(state)
        if byte >= self.boundary_offset:
            byte -= self.boundary_offset
        key = (state << _SHIFT) | byte
        nxt = self._memo.get(key)
        if nxt is None:
            nxt = self._goto(state, byte)
            self._memo[key] = nxt
        return nxt, self._answer(nxt)

    def _answer(self, state: int) -> int:
        b = self._best[state]
        if b < 0:
            raise NoMatchError("no vocabulary entry is a suffix of the history")
        return b

    def _goto(self, state: int, byte: int) -> int:
        edges, fail = self._edges, self._fail
        while True:
            nxt = edges.get((state << _SHIFT) | byte)
            if nxt is not None:
                return nxt
            if state == 0:
                return 0
            state = fail[state]

    def run(self, history: Iterable[int], state: int = 0) -> tuple[int, list[int]]:
        """Feed a whole history; returns (final_state, subword id per position)."""
        out = []
        for b in history:
            state, sid = self.step(state, b)
            out.append(sid)
        return state, out
