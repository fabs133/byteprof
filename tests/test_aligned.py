"""Patch-aligned round policies on hand-made atom sequences (no model)."""

import math

from byteprof.drafting import make_atom
from byteprof.drafting.aligned import (hybrid_draft, plan_round, simulate_policy, two_state_bytes_per_round,
                                       two_state_figure)
from byteprof.drafting.drafters import ContextDrafter
from byteprof.drafting.sim import simulate


def A(b, f=0):
    return make_atom(b, f)


class Fake:
    """Sources with scripted answers: lookups by truth index of the last emitted atom, LM drafts by position."""

    def __init__(self, raw=None, flagged=None, lm=None, lm_ignores_k=False):
        self.raw, self.flagged, self.lm = raw or {}, flagged or {}, lm or {}
        self.i = 0
        self.lm_asked = []
        self.lm_ignores_k = lm_ignores_k

    def lookup_raw(self, k):
        return list(self.raw.get(self.i, []))[:k]

    def lookup_flagged(self, k):
        return list(self.flagged.get(self.i, []))[:k]

    def lm_at(self, j, k):
        self.lm_asked.append((j, k))
        d = list(self.lm.get(j, []))
        return (d if self.lm_ignores_k else d[:k]), 1

    def push(self, atoms):
        self.i += len(atoms)


# truth: a0 mid-patch; patch ends at index 2 (flag); next patch ends at index 5
#            idx: 0     1     2         3     4     5
TRUTH = [A(10), A(11), A(12, 1), A(13), A(14), A(15, 1), A(16), A(17), A(18, 1)]


def test_p2_wrong_completion_never_counts_the_lm_part():
    # completion gets the second byte wrong, then claims a boundary; the LM part is the true continuation
    src = Fake(flagged={0: [A(11), A(99, 1)]}, lm={2: [A(13), A(14), A(15, 1)]})
    res = simulate_policy(TRUTH, "P2", src, 8, max_new=3)
    # the LM draft is asked (the drafter cost is paid) but one atom is accepted + the free atom
    assert src.lm_asked and src.lm_asked[0][0] == 2
    first = res.by_kp  # one round, draft = completion + LM part (5 atoms), only the first accepted
    assert res.rounds >= 1
    assert res.emitted == 3
    # the accepted prefix of round 1 is 1 atom (a1); a2 is the free atom: 2 atoms emitted by round 1
    assert res.start_bytes[0] == 2


def test_p2_right_completion_wrong_flag_stops_before_the_lm_part():
    # bytes right, but the draft flags a boundary one atom too early (truth a1 is unflagged)
    src = Fake(flagged={0: [A(11, 1), A(12)]}, lm={1: [A(12), A(13), A(14)]})
    plan = plan_round("P2", 0, TRUTH, src, 8)
    assert plan.lm_call and plan.completion_len == 1
    res = simulate_policy(TRUTH, "P2", Fake(flagged={0: [A(11, 1)]}, lm={1: [A(12, 1), A(13), A(14)]}), 8, max_new=1)
    # flag mismatch on the very first draft atom: L = 0, one free atom
    assert res.start_bytes[0] == 1 and res.accepted == 0


def test_p2_right_completion_and_flag_reaches_the_lm_part():
    src = Fake(flagged={0: [A(11), A(12, 1)]}, lm={2: [A(13), A(14), A(15, 1)]})
    res = simulate_policy(TRUTH, "P2", src, 8, max_new=6)
    # completion (2) + LM part (3) all accepted: 5 accepted + 1 free = 6 emitted in the first round
    assert res.rounds == 1 and res.emitted == 6
    assert res.lm_calls == 1 and res.kinds == {"within": 0, "cross": 1}


def test_p2_without_predicted_boundary_does_not_call_the_lm():
    src = Fake(flagged={0: [A(11), A(12)]}, lm={2: [A(13)]})
    plan = plan_round("P2", 0, TRUTH, src, 8)
    assert not plan.lm_call and src.lm_asked == [] and len(plan.draft) == 2


def test_cap_inside_a_hybrid_draft():
    comp = [A(11), A(12, 1)]
    lm = [A(13)] * 10
    assert len(hybrid_draft(comp, lm, 5)) == 5
    src = Fake(flagged={0: [A(11), A(12, 1)]}, lm={2: lm}, lm_ignores_k=True)
    plan = plan_round("P2", 0, TRUTH, src, 5)
    assert len(plan.draft) == 5 and plan.draft[:2] == comp
    assert src.lm_asked == [(2, 3)]  # the LM part is asked for K minus the completion


def test_no_room_for_the_lm_part_means_no_lm_call():
    src = Fake(flagged={0: [A(11), A(12, 1)]}, lm={2: [A(13)]})
    plan = plan_round("P2", 0, TRUTH, src, 2)
    assert not plan.lm_call and len(plan.draft) == 2


def test_p3_switches_between_round_types():
    src = Fake(raw={0: [A(11), A(12)]}, lm={2: [A(13), A(14), A(15, 1), A(16)]})
    res = simulate_policy(TRUTH, "P3", src, 8, max_new=7)
    # round 1: a0 mid-patch -> within-patch lookup; the patch ends at a2 (flagged p_2) so L = 1 and a2 is the free atom
    # round 2: a2 flagged -> LM crossing draft (a3 a4 a5 a6): all right, 4 + free = 5 -> total 2 + 5
    assert res.kinds == {"within": 1, "cross": 1}
    assert res.lm_calls == 1 and res.emitted == 7
    assert res.start_rounds == {0: 1, 1: 1}
    assert res.trans == {(0, 1): 1, (1, 0): 1}


def test_p3_mid_patch_round_is_single_patch_rule():
    # the lookup draft runs over the boundary with right bytes; the within-patch rule stops at the flagged p
    src = Fake(raw={0: [A(11), A(12), A(13), A(14)]})
    res = simulate_policy(TRUTH, "P3", src, 8, max_new=2)
    assert res.start_bytes[0] == 2  # a1 accepted, a2 (flagged) is the free atom


def test_p4_gates_on_the_boundary_flag_of_the_last_atom():
    src = Fake(raw={0: [A(11)]}, flagged={2: [A(13), A(14), A(15, 1)]})
    p0 = plan_round("P4", 0, TRUTH, src, 8)
    p1 = plan_round("P4", 2, TRUTH, src, 8)
    assert (p0.multi, p0.lookup_raw_call) == (False, True)
    assert (p1.multi, p1.lookup_flagged_call, p1.lm_call) == (True, True, False)


def test_p2b_uses_the_lm_directly_at_a_boundary():
    src = Fake(flagged={2: [A(13)]}, lm={2: [A(13), A(14)]})
    plan = plan_round("P2b", 2, TRUTH, src, 8)
    assert plan.lm_call and len(plan.draft) == 2 and not plan.lookup_flagged_call


def test_oracle_completion_is_exact_and_still_uses_the_real_lm_part():
    src = Fake(lm={2: [A(13), A(14)]})
    plan = plan_round("P2", 0, TRUTH, src, 8, oracle=True)
    assert plan.draft[:2] == [A(11), A(12, 1)] and plan.lm_call
    res = simulate_policy(TRUTH, "P3", Fake(lm={2: [A(13)]}), 8, max_new=3, oracle=True)
    assert res.kinds["within"] >= 1


def test_p0_and_p1_agree_with_sim_simulate():
    pattern = [A(1), A(2), A(3, 1), A(4), A(5, 1)]
    truth = pattern * 12
    for multi, pol in ((False, "P0"), (True, "P1")):
        for k in (1, 3, 8):
            ref = simulate(truth, ContextDrafter(min_len=2), k, ctx_atoms=pattern * 2, max_new=40, multi_patch=multi,
                           max_patches=8 if multi else None)

            class S(Fake):
                def __init__(self):
                    super().__init__()
                    self.d = ContextDrafter(min_len=2)
                    self.d.reset(pattern * 2 + [truth[0]], truth)

                def lookup_raw(self, k_):
                    return self.d.propose(k_)

                lookup_flagged = lookup_raw

                def push(self, atoms):
                    self.d.push(atoms)

            got = simulate_policy(truth, pol, S(), k, max_new=40)
            assert (got.emitted, got.rounds, got.global_passes) == (ref.emitted, ref.rounds, ref.global_passes), (pol, k)
            assert got.by_kp.keys() and sum(got.by_kp.values()) == got.rounds


def test_two_state_model_is_exact_for_a_deterministic_chain():
    # rounds alternate mid -> boundary -> mid ...: half of the rounds in each state
    tr = {(0, 1): 10, (1, 0): 10}
    assert math.isclose(two_state_bytes_per_round(tr, {0: 2.0, 1: 10.0}), 6.0)
    # absorbing mid: all rounds mid
    assert math.isclose(two_state_bytes_per_round({(0, 0): 5, (1, 0): 5}, {0: 2.0, 1: 10.0}), 2.0)


def test_two_state_model_reproduces_pooled_rounds_from_a_simulation():
    src = Fake(raw={0: [A(11), A(12)], 3: [A(14), A(15)]})
    res = simulate_policy(TRUTH, "P0", src, 8, max_new=8)
    per = {s: res.start_bytes[s] / res.start_rounds[s] for s in (0, 1) if res.start_rounds[s]}
    per.setdefault(0, 0.0)
    per.setdefault(1, 0.0)
    got = two_state_bytes_per_round({k: float(v) for k, v in res.trans.items()}, per)
    assert abs(got - res.bytes_per_round) <= 0.1 * res.bytes_per_round + 1e-9


def test_two_state_figure_limits():
    # perfect drafter: every round fills the cap
    assert math.isclose(two_state_figure(1.0, 1.0, 0.5, 1, 1, k=32), 33.0)
    # hit rate 0: only the free unit
    assert math.isclose(two_state_figure(0.0, 0.0, 0.3, 4, 1), 0.3 * 4 + 0.7 * 1)
    # a token free unit that always restarts at a boundary beats a byte free unit at the same hit rates
    assert two_state_figure(0.9, 0.3, 1.0, 4, 4) > two_state_figure(0.9, 0.3, 0.32, 1, 1)


def test_l1_asks_the_lm_every_round_with_the_crossing_rule():
    src = Fake(lm={0: [A(11), A(12, 1), A(13)], 4: [A(15, 1)]})
    res = simulate_policy(TRUTH, "L1", src, 8, max_new=5)
    assert res.lm_calls == res.rounds and res.kinds["within"] == 0
    assert res.by_kp and all(k[3] for k in res.by_kp)


def test_patch_count_includes_the_patch_of_a0():
    truth = TRUTH[2:]  # a0 is flagged: the round starts a new patch
    src = Fake(flagged={0: [A(13), A(14), A(15, 1)]})
    res = simulate_policy(truth, "P1", src, 8, max_new=4)
    assert (3, 2, True, False) in res.by_kp  # a0's patch + the flagged draft atom
    src = Fake(raw={0: [A(13), A(14)]})
    res = simulate_policy(truth, "P0", src, 8, max_new=3)
    assert (2, 1, False, False) in res.by_kp


# ---- TASK_build_p4.md: P4 with a separate K for the mid-patch rounds (`k_within`) ----------------------------------


def test_p4_k_within_only_shortens_mid_patch_drafts():
    raw = {0: [A(11), A(12, 1), A(13), A(14)], 2: [A(13), A(14), A(15, 1), A(16)]}
    flagged = {2: [A(13), A(14), A(15, 1), A(16)]}
    src = Fake(raw=raw, flagged=flagged)
    assert len(plan_round("P4", 0, TRUTH, src, 8).draft) == 4
    short = plan_round("P4", 0, TRUTH, src, 8, k_within=2)
    assert len(short.draft) == 2 and not short.multi
    # a round that starts at a boundary (truth[2] is flagged) uses the crossing K, whatever k_within says
    src.i = 2  # the scripted lookups are keyed by the index of the last emitted atom
    cross = plan_round("P4", 2, TRUTH, src, 8, k_within=2)
    assert cross.multi and len(cross.draft) == 4


def test_p4_k_within_none_is_the_old_policy():
    raw = {0: [A(11), A(12, 1), A(13)], 3: [A(14), A(15, 1), A(16)]}
    flagged = {2: [A(13), A(14), A(15, 1)]}
    a = simulate_policy(TRUTH, "P4", Fake(raw=raw, flagged=flagged), 8, max_new=8)
    b = simulate_policy(TRUTH, "P4", Fake(raw=raw, flagged=flagged), 8, max_new=8, k_within=None)
    assert (a.emitted, a.rounds, a.by_kp, a.global_passes) == (b.emitted, b.rounds, b.by_kp, b.global_passes)
    c = simulate_policy(TRUTH, "P4", Fake(raw=raw, flagged=flagged), 8, max_new=8, k_within=1)
    assert max(kp[0] for kp, in [(k,) for k in c.by_kp if not k[2]]) <= 1  # within-patch rounds never carry more than k_within atoms
