"""Drafters (context lookup, n-gram, combined, oracle) and the corpus overlap checker."""

import numpy as np

from byteprof.drafting import byte_of, flag_of, make_atom as A
from byteprof.drafting.corpus import shared_windows
from byteprof.drafting.drafters import CombinedDrafter, ContextDrafter, OracleDrafter
from byteprof.drafting.ngram import NGramDrafter, NGramModel


def atoms(s: bytes, flags=()):
    return [A(b, i in flags) for i, b in enumerate(s)]


def test_context_proposes_the_continuation_of_the_longest_match():
    d = ContextDrafter(min_len=3)
    d.reset(atoms(b"xx the cat sat. the cat"))
    p = d.propose(5)
    assert bytes(byte_of(a) for a in p) == b" sat."


def test_context_prefers_longer_match_over_more_recent_short_one():
    d = ContextDrafter(min_len=2)
    d.reset(atoms(b"abcXY1 zzXY2 abcXY"))
    # suffix 'abcXY' occurred at the start (continuation '1'), 'cXY'/'XY' also later (continuation '2')
    assert bytes(byte_of(a) for a in d.propose(1)) == b"1"
    assert d.last_match == 5


def test_context_min_len_and_no_match():
    d = ContextDrafter(min_len=4)
    d.reset(atoms(b"abc abc"))
    assert d.propose(3) == []  # only 3 bytes ("abc") match, below min_len 4
    d2 = ContextDrafter(min_len=3)
    d2.reset(atoms(b"abc abc"))
    assert bytes(byte_of(a) for a in d2.propose(2)) == b" a"


def test_context_copies_flags_and_extends_periodically():
    d = ContextDrafter(min_len=2)
    d.reset(atoms(b"ababab", flags={3}))
    p = d.propose(5)
    # match for suffix 'ab' ends at index 3 (flagged b): continuation a(4) b(5) then periodic a b a
    assert [byte_of(a) for a in p] == list(b"ababa"[:5])
    assert flag_of(p[0]) == 0
    d2 = ContextDrafter(min_len=2)
    d2.reset(atoms(b"xyzwxy", flags={2}))
    p2 = d2.propose(2)
    assert [byte_of(a) for a in p2] == list(b"zw") and flag_of(p2[0]) == 1


def test_context_push_updates_the_index():
    d = ContextDrafter(min_len=3)
    d.reset(atoms(b"hello wor"))
    assert d.propose(3) == []
    d.push(atoms(b"ld, hello wor"))
    assert bytes(byte_of(a) for a in d.propose(2)) == b"ld"


def test_oracle_returns_true_future_and_follows_pushes():
    truth = atoms(b"0123456789")
    o = OracleDrafter()
    o.reset([], truth)
    assert o.propose(3) == truth[1:4]
    o.push(truth[1:3])
    assert o.propose(2) == truth[3:5]
    assert o.propose(100) == truth[3:]


def _small_model(text: bytes, flags: np.ndarray, **kw):
    return NGramModel.build(text, flags, orders=(1, 2, 3), bits=12, **kw)


def test_ngram_greedy_continuation_and_flags():
    text = b"hello world " * 50
    fl = np.zeros(len(text), np.uint8)
    for i, b in enumerate(text):
        if b == ord("o"):
            fl[i] = 1
    m = _small_model(text, fl)
    d = NGramDrafter(m)
    d.reset(atoms(b"say hel"))
    p = d.propose(6)
    assert bytes(byte_of(a) for a in p) == b"lo wor"
    assert flag_of(p[1]) == 1 and flag_of(p[0]) == 0  # 'o' is flagged in the corpus


def test_ngram_min_count_and_unknown_context():
    text = b"abcd" * 20 + b"xyz"
    m = _small_model(text, np.zeros(len(text), np.uint8))
    assert m.next_atom([ord(c) for c in "abc"], min_count=1) == ord("d")
    assert m.next_atom([ord(c) for c in "xyz"], min_count=1) == -1  # never followed by anything
    assert m.next_atom([ord(c) for c in "abc"], min_count=200) == -1


def test_combined_uses_context_first_then_ngram():
    text = b"the quick brown fox " * 30
    m = _small_model(text, np.zeros(len(text), np.uint8))
    c = CombinedDrafter(ContextDrafter(min_len=4), NGramDrafter(m))
    c.reset(atoms(b"zzzz qqqq the quic"))
    assert bytes(byte_of(a) for a in c.propose(3)) == b"k b"
    assert c.used_ngram == 1 and c.used_context == 0  # no 4-byte earlier match: n-gram answered
    c.push(atoms(b"k brown fox zzzz qqqq"))
    p = c.propose(2)
    assert c.used_context == 1


def test_shared_windows_finds_exact_overlaps_only():
    corpus = b"0123456789" * 10 + b"UNIQUE-PROMPT-TEXT-THAT-IS-LONG-ENOUGH-FOR-A-WINDOW"
    prompt_in = b"xx" + b"UNIQUE-PROMPT-TEXT-THAT-IS-LONG-ENOUGH-FOR-A-WINDOW"[:40]
    prompt_out = b"completely different prompt text that shares nothing at all, really"
    hits = shared_windows(corpus, {"in": prompt_in, "out": prompt_out}, w=32)
    assert hits and all(h[0] == "in" for h in hits)
    assert not shared_windows(corpus, {"out": prompt_out}, w=32)
    assert not shared_windows(corpus, {"short": b"short"}, w=32)
