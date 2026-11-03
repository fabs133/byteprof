"""Round-rule simulator on hand-made atom sequences (TASK_drafting.md, "The round rule")."""

from byteprof.drafting import make_atom as A
from byteprof.drafting.drafters import NullDrafter, OracleDrafter
from byteprof.drafting.sim import accepted_prefix, simulate


class Fixed:
    """Proposes a fixed atom list every round (ignores the context)."""

    def __init__(self, atoms):
        self.atoms = list(atoms)

    def reset(self, ctx_atoms, truth=None):
        pass

    def propose(self, k):
        return self.atoms[:k]

    def push(self, atoms):
        pass


def seq(n, flags=()):
    """a_0 .. a_n with bytes 1..n+1; atoms at the indices in `flags` carry the boundary flag."""
    return [A(1 + i, i in flags) for i in range(n + 1)]


def test_k0_is_one_atom_per_round():
    t = seq(10)
    r = simulate(t, OracleDrafter(), 0)
    assert (r.rounds, r.emitted, r.bytes_per_local_pass) == (10, 10, 1.0)
    assert r.l_hist == {0: 10}


def test_k0_with_none_and_flags_counts_global_passes():
    t = seq(10, flags={2, 5, 9})
    r = simulate(t, None, 0)
    assert r.rounds == 10
    # a0 flagged at rounds that start after a_2, a_5 (a_9's round is the 10th round: a_9 is a0 of round 10)
    assert r.global_passes == 3
    assert r.bytes_per_global_pass == 10 / 3


def test_full_acceptance_oracle():
    t = seq(10)
    r = simulate(t, OracleDrafter(), 4)
    assert r.rounds == 2 and r.emitted == 10 and r.l_hist == {4: 2}
    assert r.bytes_per_local_pass == 5.0


def test_first_byte_mismatch_emits_one():
    t = seq(8)
    wrong = [A(200, 0)] * 4
    r = simulate(t, Fixed(wrong), 4)
    assert r.l_hist == {0: 8} and r.rounds == 8


def test_partial_acceptance_then_mismatch():
    t = seq(6)  # a_1..a_6 = bytes 2..7
    draft = [A(2, 0), A(3, 0), A(99, 0), A(5, 0)]  # third draft wrong
    # round from a_0: d1==p1, d2==p2, d3!=p3 -> L=2 -> emits p1..p3 = 3 atoms
    r = simulate(t, Fixed(draft), 4, max_new=3)
    assert r.rounds == 1 and r.emitted == 3 and r.l_hist == {2: 1}


def test_boundary_in_the_draft_ends_the_round():
    t = seq(10, flags={3})  # a_3 carries the flag
    r = simulate(t, OracleDrafter(), 8)
    # round 1: p1, p2 unflagged, p3 flagged -> L=2 (d3 may not be accepted since p3 is flagged) -> 3 atoms
    assert r.l_hist[2] == 1
    first = accepted_prefix([t[i] for i in range(1, 9)], t[1:10], 8, multi_patch=False)
    assert first == 2


def test_flagged_atom_is_still_emitted():
    t = seq(4, flags={1})
    r = simulate(t, OracleDrafter(), 4, max_new=4)
    # round 1: p1 flagged: L=0, emits 1; round 2 starts with a0 flagged (global runs); 3 atoms remain: L=3, emits 3
    assert r.emitted == 4 and r.rounds == 2 and r.global_passes == 1 and r.l_hist == {0: 1, 3: 1}


def test_cap_truncates_emission_exactly():
    t = seq(10)
    r = simulate(t, OracleDrafter(), 8)
    assert r.emitted == 10 and r.rounds == 2 and r.l_hist == {8: 1, 1: 1}
    r = simulate(t, OracleDrafter(), 8, max_new=7)
    assert r.emitted == 7 and r.rounds == 1


def test_multi_patch_variant_does_not_stop_at_flags_and_needs_flag_agreement():
    t = seq(10, flags={3, 7})
    s = simulate(t, OracleDrafter(), 8, multi_patch=False)
    m = simulate(t, OracleDrafter(), 8, multi_patch=True)
    assert s.rounds > m.rounds
    assert m.rounds == 2 and m.emitted == 10
    # flags in the draft that disagree with the truth stop acceptance
    wrong_flag = [A(2, 1), A(3, 0)]
    assert accepted_prefix(wrong_flag, t[1:], 2, multi_patch=True) == 0
    assert accepted_prefix(wrong_flag, t[1:], 2, multi_patch=False) == 2
    # global passes: one per round that consumed a flagged atom
    assert m.global_passes == 1  # round 1 consumed positions 0..8 which include flagged a_3 and a_7; round 2 none


def test_global_pass_counted_when_a0_flagged_multi_patch():
    t = seq(6, flags={0})
    r = simulate(t, NullDrafter(), 0, multi_patch=True)
    assert r.global_passes == 1


def test_available_graph_sizes_round_the_draft_down():
    t = seq(40)
    r = simulate(t, OracleDrafter(), 8, ks=(0, 2, 4), max_new=20)
    assert set(r.by_k) == {4}  # an 8-atom draft is cut to the largest available size not above 8
    assert r.l_hist == {4: 4} and r.emitted == 20
    class Short(Fixed):
        def propose(self, k):
            return self.atoms[:3]
    r = simulate(t, Short([A(2, 0), A(3, 0), A(4, 0)]), 8, ks=(0, 2, 4), max_new=6)
    assert set(r.by_k) <= {0, 2}  # a 3-atom draft runs as K = 2
    assert sum(r.by_k.values()) == r.rounds


def test_never_emits_more_than_available_truth():
    t = seq(5)
    r = simulate(t, OracleDrafter(), 8, max_new=100)
    assert r.emitted == 5
