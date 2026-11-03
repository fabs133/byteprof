"""Drafter diagnostics: round-end classifier, replay vs the simulators, cross-simulator agreement (no model)."""

import random

from byteprof.drafting import flag_of, make_atom
from byteprof.drafting.diagnostics import (classify_end, classify_position, next_token_agreement, replay_byte_rounds,
                                           replay_token_rounds, ws_near_miss)
from byteprof.drafting.sim import simulate
from byteprof.drafting.tokens import simulate_tokens


def A(text: bytes, flags):
    return [make_atom(b, f) for b, f in zip(text, flags)]


# --- round-end classifier ---------------------------------------------------------------------------------

def test_classify_end_each_kind():
    t = A(b"abcdef", [0, 0, 1, 0, 0, 1])
    k = 8
    assert classify_end(t[:3], t, k, True) == (3, "exhausted")  # all accepted, drafter had nothing more
    assert classify_end(t[:3], t, k, True, more_available=True) == (3, "cap")
    assert classify_end(t[:2] + [make_atom(ord("x"), 1)], t, k, True) == (2, "wrong_byte")
    assert classify_end(t[:2] + [make_atom(ord("c"), 0)], t, k, True) == (2, "wrong_flag")  # byte right, flag wrong
    assert classify_end(t[:2] + [make_atom(ord("c"), 1)], t, k, True) == (3, "exhausted")
    assert classify_end(t[:4], t, 4, True) == (4, "cap")  # draft reached K
    assert classify_end(t[:3], t, k, True, patch_truncated=True) == (3, "patch_cap")
    assert classify_end(t[:3], t, k, True, room=2) == (3, "end_of_generation")
    # single-patch rule: a flagged prediction ends the round even though the byte matches
    assert classify_end(t[:4], t, k, False) == (2, "patch_end")


def test_classify_end_agrees_with_accepted_prefix_on_random_cases():
    from byteprof.drafting.sim import accepted_prefix

    rng = random.Random(1)
    for _ in range(300):
        n = rng.randint(1, 12)
        truth = [make_atom(rng.randint(97, 99), rng.random() < 0.3) for _ in range(n)]
        draft = [a if rng.random() < 0.8 else make_atom(rng.randint(97, 99), rng.random() < 0.3) for a in truth][:rng.randint(0, min(n, 8))]
        for multi in (True, False):
            L, end = classify_end(draft, truth, 8, multi)
            assert L == accepted_prefix(draft, truth, 8, multi)
            if L < len(draft):
                assert end in ("wrong_byte", "wrong_flag", "patch_end")
                if end == "wrong_byte":
                    assert (draft[L] & 255) != (truth[L] & 255)
                else:  # bytes equal: only a flag can end the round
                    assert (draft[L] & 255) == (truth[L] & 255)


class Oracle:
    """Byte drafter that proposes the true continuation, wrong from `wrong_at` bytes on (per round)."""

    def __init__(self, text: bytes, flags, miss_every: int = 5):
        self.atoms = A(text, flags)
        self.miss_every = miss_every
        self.i = 0
        self.n = 0

    def reset(self, ctx, truth=None):
        self.i = 0
        self.n = 0

    def propose(self, k):
        d = list(self.atoms[self.i + 1: self.i + 1 + k])
        if k < (1 << 19) and len(d) > self.miss_every:  # introduce a wrong byte at a fixed place
            d[self.miss_every] = make_atom((d[self.miss_every] & 255) ^ 1, flag_of(d[self.miss_every]))
        return d

    def push(self, atoms):
        self.i += len(atoms)


def test_replay_byte_rounds_reproduces_simulate():
    text = b"the cat sat on the mat and then it left the house"
    flags = [1 if (j + 1 == len(text) or text[j + 1] == 32) else 0 for j in range(len(text))]
    truth = A(text, flags)
    for k in (4, 8, 16):
        for multi in (True, False):
            for mp in (None, 2):
                recs = replay_byte_rounds(truth, Oracle(text, flags), k, multi_patch=multi, max_patches=mp if multi else None)
                res = simulate(truth, Oracle(text, flags), k, multi_patch=multi, max_patches=mp if multi else None)
                assert len(recs) == res.rounds and sum(r.emitted for r in recs) == res.emitted
                hist = {}
                for r in recs:
                    hist[r.L] = hist.get(r.L, 0) + 1
                assert hist == {a: b for a, b in res.l_hist.items()}
    recs = replay_byte_rounds(truth, Oracle(text, flags), 16, multi_patch=True)
    assert any(r.end == "wrong_byte" for r in recs)
    wr = next(r for r in recs if r.end == "wrong_byte")
    assert wr.miss_truth[:1] != wr.miss_draft[:1] and wr.miss_i == wr.i + 1 + wr.L


# --- the two simulators on one identical case ---------------------------------------------------------------

class TokDrafter:
    """Token drafter proposing the true next tokens, wrong at a fixed place."""

    def __init__(self, toks, wrong_at):
        self.toks, self.wrong_at, self.i = toks, wrong_at, 0

    def reset(self, ctx, truth=None):
        self.i = 0

    def propose(self, k):
        d = list(self.toks[self.i + 1: self.i + 1 + k])
        if k < (1 << 19) and len(d) > self.wrong_at:
            d[self.wrong_at] = -1
        return d

    def push(self, toks):
        self.i += len(toks)


def test_byte_and_token_rule_agree_when_every_token_is_one_flagged_byte():
    """If every token is a single byte (so every byte carries a flag), the bonus unit, the accepted prefix, K and
    the round count are the same in both rules: they must give identical rounds and emitted counts."""
    text = b"abcabcabcabcabcabcabcabcabc"
    truth_b = A(text, [1] * len(text))
    toks = list(text)
    for k in (3, 6):
        for wrong in (1, 2, 4):
            rb = replay_byte_rounds(truth_b, Oracle(text, [1] * len(text), wrong), k, multi_patch=True)
            rt = replay_token_rounds(toks, TokDrafter(toks, wrong), k, lambda t: 1)
            assert [(r.draft_len, r.L, r.emitted) for r in rb] == [(r.draft_len, r.L, r.emitted) for r in rt]
            assert sum(r.nbytes for r in rb) == sum(r.nbytes for r in rt) == len(text) - 1
            sim_t = simulate_tokens(toks, TokDrafter(toks, wrong), k)
            assert sim_t.rounds == len(rt) and sim_t.emitted == sum(r.emitted for r in rt)


def test_token_rule_credits_the_bonus_token_in_bytes():
    """Three-byte tokens: the same accepted units give three times the bytes in the token rule's bonus unit."""
    toks = [10, 11, 12, 13, 14, 15, 16, 17]
    rt = replay_token_rounds(toks, TokDrafter(toks, 99), 2, lambda t: 3)
    assert sum(r.nbytes for r in rt) == 3 * sum(r.emitted for r in rt) == 3 * (len(toks) - 1)
    # every round emits one bonus token (3 bytes) on top of its accepted tokens
    assert all(r.nbytes == 3 * (1 + r.L) for r in rt[:-1])  # the last round may be cut by the end of generation


def test_token_rule_end_reasons():
    toks = list(range(20))
    rt = replay_token_rounds(toks, TokDrafter(toks, 2), 4, lambda t: 1)
    assert rt[0].end == "wrong_token" and rt[0].L == 2 and rt[0].miss_i == 3
    rt = replay_token_rounds(toks, TokDrafter(toks, 99), 4, lambda t: 1)
    assert rt[0].end == "cap" and rt[0].L == 4
    rt = replay_token_rounds(toks, TokDrafter(toks, 99), 4, lambda t: 1, max_new=3)
    assert rt[0].end == "end_of_generation" and rt[0].emitted == 3


# --- position / near-miss / agreement ---------------------------------------------------------------------

def test_classify_position():
    s = "ab cd, straße".encode()
    assert classify_position(s, 0) == "word_start" and classify_position(s, 1) == "inside_word"
    assert classify_position(s, 2) == "whitespace_punct" and classify_position(s, 5) == "whitespace_punct"
    assert classify_position(s, s.index("ß".encode())) == "multibyte"
    assert classify_position(s, s.index("ß".encode()) + 2) == "inside_word"


def test_ws_near_miss():
    assert ws_near_miss(b"  the cat sat", b" the cat sat on")  # same words, different spaces
    assert not ws_near_miss(b"a dog sat", b"the cat sat")
    assert not ws_near_miss(b"the cat", b"the cat")  # identical is not a miss
    assert not ws_near_miss(b" a", b"a")  # too little evidence


def test_next_token_agreement_up_to_shorter_length():
    assert next_token_agreement(b" the", b" theory") == (True, False)
    assert next_token_agreement(b" the", b" the") == (True, True)
    assert next_token_agreement(b" a", b" the") == (False, False)
    assert next_token_agreement(b"", b"x") == (False, False)


def test_context_back_space_keeps_the_leading_space_with_the_fragment():
    from byteprof.drafting.diagnostics import context_back_space
    from byteprof.drafting.lmdraft import context_back

    assert context_back_space(b"a strong advoc") == len(b" advoc")
    assert context_back(b"a strong advoc", True) == len(b"advoc")  # Task B's healing leaves the space in the context
    assert context_back_space(b"a strong ") == 1  # a lone trailing space is held back as well
    assert context_back_space(b"lighthouse,") == 0
    assert context_back_space(b"x\n") == 0
    assert context_back_space("grü".encode()[:-1]) == len("gr") + 1  # incomplete character + fragment


def test_align_extends_the_bonus_to_the_token_end_and_matches_the_token_rule_on_whole_token_drafts():
    """With `align`, a byte round that stops inside a token also emits the rest of that token: every round (but the
    last) ends on a token end, all bytes are still emitted, and no more rounds are needed than without `align`."""
    words = [b"abc ", b"de ", b"fghij ", b"kl ", b"mnopq ", b"rs ", b"tuv ", b"wxyz "]
    text = b"".join(words)
    ends = []
    pos = 0
    for w in words:
        pos += len(w)
        ends.append(pos)
    flags = [1 if (j + 1) in ends else 0 for j in range(len(text))]
    truth_b = A(text, flags)  # truth[0] = first byte: start after the first token instead
    base = ends[0]
    truth = truth_b[base - 1:]  # a_0 = last byte of the first token
    bnd = {e - base for e in ends}  # truth indices that are token ends
    toks = list(range(1, len(words)))  # token ids of the remaining words, truth[0] = token 0 (the first word)
    toks = [0] + toks

    class ByteOracle:
        def __init__(self):
            self.i = 0

        def reset(self, ctx, t=None):
            self.i = 0

        def propose(self, k):
            d = list(truth[self.i + 1: self.i + 1 + k])
            if k < (1 << 19) and len(d) > 5:  # a wrong byte inside the 2nd word of the draft
                d[5] = make_atom((d[5] & 255) ^ 1, flag_of(d[5]))
            return d

        def push(self, atoms):
            self.i += len(atoms)

    plain = replay_byte_rounds(truth, ByteOracle(), 8, multi_patch=True)
    aligned = replay_byte_rounds(truth, ByteOracle(), 8, multi_patch=True, align=bnd)
    assert sum(r.emitted for r in aligned) == len(truth) - 1
    assert len(aligned) <= len(plain)
    assert all((r.i + r.emitted) in bnd for r in aligned[:-1])  # every round ends on a token end


class Empty:
    def reset(self, ctx, truth=None):
        pass

    def propose(self, k):
        return []

    def push(self, atoms):
        pass


def test_align_with_no_draft_emits_exactly_one_token_per_round():
    words = [b"abc ", b"x ", b"fghij ", b"k ", b"mn ", b"o "]
    text = b"".join(words)
    ends = [sum(map(len, words[:j + 1])) for j in range(len(words))]
    truth = A(text, [1 if (j + 1) in ends else 0 for j in range(len(text))])[ends[0] - 1:]
    bnd = {e - ends[0] for e in ends}
    recs = replay_byte_rounds(truth, Empty(), 8, multi_patch=True, align=bnd)
    assert [r.emitted for r in recs] == [len(w) for w in words[1:]]  # each round completes one token
    assert [r.emitted for r in replay_byte_rounds(truth, Empty(), 8, multi_patch=True)] == [1] * (len(truth) - 1)


def test_patch_limit_ends_the_round_as_patch_cap():
    text = b"ab cd ef gh ij kl"
    flags = [1 if (j + 1 == len(text) or text[j + 1] == 32) else 0 for j in range(len(text))]
    truth = A(text, flags)

    class Perfect:
        def reset(self, ctx, t=None):
            self.i = 0

        def propose(self, k):
            return list(truth[self.i + 1: self.i + 1 + k])

        def push(self, atoms):
            self.i += len(atoms)

    recs = replay_byte_rounds(truth, Perfect(), 16, multi_patch=True, max_patches=2)
    assert recs[0].end == "patch_cap" and recs[0].L == recs[0].draft_len
    free = replay_byte_rounds(truth, Perfect(), 16, multi_patch=True)
    assert len(recs) > len(free)
