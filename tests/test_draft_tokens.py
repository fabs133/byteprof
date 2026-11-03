"""Token drafters and the token round simulator (OLMo 2 side of the drafting task)."""

import numpy as np

from byteprof.drafting.drafters import CombinedDrafter, OracleDrafter
from byteprof.drafting.tokens import (TokenContextDrafter, TokenNGramDrafter, TokenNGramModel, simulate_tokens)


def test_token_context_drafter_copies_the_longest_earlier_continuation():
    d = TokenContextDrafter(min_len=2)
    d.reset([5, 6, 7, 8, 9, 1, 2, 5, 6, 7])
    assert d.propose(3) == [8, 9, 1]
    d.push([8, 9])
    assert d.propose(2) == [1, 2]
    assert TokenContextDrafter(min_len=3).propose(2) == []


def test_token_context_drafter_prefers_the_longer_match_and_extends_periodically():
    d = TokenContextDrafter(min_len=1)
    d.reset([1, 2, 3, 9, 4, 3, 7, 1, 2, 3])  # suffix (1,2,3) matches at the start (continuation 9), (3,) later (7)
    assert d.propose(1) == [9] and d.last_match == 3
    p = TokenContextDrafter(min_len=2)
    p.reset([4, 5, 4, 5])
    assert p.propose(5) == [4, 5, 4, 5, 4]


def test_token_ngram_greedy_continuation_and_min_count():
    toks = np.array(([10, 20, 30, 40] * 50) + [99], np.int64)
    m = TokenNGramModel.build(toks, orders=(1, 2, 3), bits=12)
    d = TokenNGramDrafter(m)
    d.reset([7, 10, 20])
    assert d.propose(4) == [30, 40, 10, 20]
    assert m.next_token([10, 20], min_count=1) == 30
    assert m.next_token([10, 20], min_count=1000) == -1
    assert m.next_token([99], min_count=1) == -1  # 99 is never followed by anything


def test_token_simulator_matches_the_round_rule():
    truth = list(range(100, 112))  # t0 .. t11
    r = simulate_tokens(truth, OracleDrafter(), 4, max_new=11)
    assert r.emitted == 11 and r.rounds == 3 and r.l_hist == {4: 2, 1: 1}  # 5 + 5 + 1 tokens: the last draft has one token left

    class Wrong:
        def reset(self, ctx, truth=None): pass
        def propose(self, k): return [0] * k
        def push(self, atoms): pass

    r = simulate_tokens(truth, Wrong(), 4, max_new=6)
    assert r.l_hist == {0: 6} and r.rounds == 6
    r = simulate_tokens(truth, OracleDrafter(), 8, ks=(0, 2, 4), max_new=11)
    assert set(r.by_k) <= {0, 4, 2}


def test_combined_token_drafter_falls_back_to_the_ngram():
    toks = np.array([1, 2, 3, 4, 5] * 40, np.int64)
    m = TokenNGramModel.build(toks, orders=(1, 2), bits=12)
    c = CombinedDrafter(TokenContextDrafter(min_len=2), TokenNGramDrafter(m))
    c.reset([9, 9, 1, 2])
    assert c.propose(3) == [3, 4, 5] and c.used_ngram == 1
    c.push([3, 4, 5, 1, 2])
    assert c.propose(2) == [3, 4] and c.used_context == 1
