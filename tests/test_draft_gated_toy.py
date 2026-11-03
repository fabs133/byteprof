"""Toy-model exactness test of boundary-gated (multi-patch) drafting rounds (CPU, fp32, nothing to download).

Same toy model as tests/test_draft_toy.py, plus an engine whose crossing round pools the encoder outputs at the flagged
inputs, runs the recurrent "global" layer over all speculative patches, gives every decoder position the vector of its
patch, accepts on raw ids (byte and flag) and rolls the global state back to the accepted patch count. The host loop,
the plan (patch select, pool positions, raw draft ids) and the device pieces (`select_patch_values`, `rollback_index`,
`patches_kept`, `accepted_raw`) are the production code of byteprof.optim.draft_gated.

Covered: wrong flags, wrong bytes after a correct boundary, more patches than slots (the patch cap and the free slots of
the global cache), the output cap, EOS inside a crossing round, and injected bugs (each must make a scenario fail).
"""

import random

import pytest
import torch

from byteprof.optim import draft_core as core
from byteprof.optim import draft_gated as gated
from test_draft_toy import (BOUNDARY, BYTE_OFFSET, D, EOS_IDS, NB, ListDrafter, ToyAutomaton, ToyEngine, ToyModel,
                            atoms_of, plain_decode, strip, toy_step)

KS = tuple(range(0, 9))
RAW0S = (4 + 3, 4 + 5 + BOUNDARY)  # a0 unflagged and flagged
VERIFY = ["steps"]


@pytest.fixture(params=["steps", "lazy"], autouse=True)
def verify_path(request):
    VERIFY[0] = request.param
    yield request.param
    VERIFY[0] = "steps"


class ToyCrossEngine(ToyEngine):
    """RoundEngine with `run_cross`; tracks the patches in the "global cache" (pos) like glob.pos."""

    def __init__(self, model, ks=KS, verify="steps", capacity=10 ** 9):
        super().__init__(model, ks, verify)
        self.pos, self.capacity = 0, capacity
        self.max_p = 0
        self.cross_rounds = 0

    def run_round(self, k, boundary, ids, subs):
        out = super().run_round(k, boundary, ids, subs)
        self.pos += int(boundary)
        assert self.pos <= self.capacity
        return out

    def run_cross(self, k, ids, raw_ids, subs, psel, pool):
        m, s = self.m, k + 1
        n_p = len(pool)
        assert 2 <= n_p <= gated.PMAX and self.pos + n_p <= self.capacity  # overflow = device-side assert on the GPU
        assert len(ids) == len(raw_ids) == len(subs) == len(psel) == s
        self.max_p = max(self.max_p, n_p)
        self.cross_rounds += 1
        lazy = self.verify == "lazy"
        enc_saved, dec_saved = ([], []) if lazy else (None, None)
        h = m.byte_emb(torch.tensor(ids).view(1, s)) + m.sub_emb(torch.tensor(subs).view(1, s))
        h = m.enc.multi(h, s, m.enc_state, self.enc_hist, self.enc_buf, enc_saved)
        rows = m.out_proj(h[0].index_select(0, torch.tensor(pool)))  # (P, D)
        g, gs = m.g, []
        for r in range(n_p):  # the recurrent "global model" over the speculative patches
            g = torch.tanh(m.glob_w(torch.cat([g, rows[r:r + 1]], dim=-1)))
            gs.append(g)
        big_g = torch.cat(gs, 0)
        psel_t = torch.tensor(psel)
        vals = gated.select_patch_values(big_g, psel_t)
        h = m.dec.multi(m.in_proj(h) + vals.view(1, s, D), s, m.dec_state, self.dec_hist, self.dec_buf, dec_saved)
        tok = torch.argmax(m.lm_head(m.norm(h)), dim=-1)[0]
        length = gated.accepted_raw(torch.tensor(raw_ids), tok, s)
        if lazy:
            core.lazy_commit_state(toy_step, m.enc_state, enc_saved[0], length, *enc_saved[1:])
            core.lazy_commit_state(toy_step, m.dec_state, dec_saved[0], length, *dec_saved[1:])
        else:
            core.commit_state(self.enc_hist, m.enc_state, length)
            core.commit_state(self.dec_hist, m.dec_state, length)
        m.g = big_g.index_select(0, gated.rollback_index(psel_t, length))
        m.last_value = m.g.view(1, 1, D).clone()
        self.pos += int(gated.patches_kept(psel_t, length))
        return tok.tolist(), int(length)


def run_gated(model, raw0, n_new, drafter, *, ks=KS, k_max=8, k_cross=8, max_patches=8, capacity=10 ** 9, eos_ids=EOS_IDS,
              stats=None, rounds=None, flagger=None, pos0=0):
    model.reset()
    eng = ToyCrossEngine(model, ks, VERIFY[0], capacity)
    aut = ToyAutomaton()
    state, sub = aut.step(0, raw0)
    stop_len = 1 + n_new
    tokens = [0] * (stop_len + 2)
    drafter.reset([core.atom_of(raw0, BYTE_OFFSET, BOUNDARY)], None)
    eng.pos = pos0
    ctx = bytearray([strip(raw0) - BYTE_OFFSET])

    def on_round(info):
        assert info["pos"] == eng.pos, (info["pos"], eng.pos)  # the host mirror of the global cache position
        if rounds is not None:
            rounds.append(info)

    n_tok = gated.gated_rounds_loop(
        eng, aut, drafter, state=state, sub=sub, raw=raw0, tokens=tokens, n_tok=1, stop_len=stop_len, eos_ids=eos_ids,
        bpe_end=-1, byte_offset=BYTE_OFFSET, boundary_offset=BOUNDARY, k_max=k_max, k_cross=k_cross,
        max_patches=max_patches, pos=pos0, capacity=capacity, ctx=ctx, flagger=flagger, stats=stats, on_round=on_round)
    return tokens[1:n_tok], eng


@pytest.fixture(scope="module")
def models():
    return [ToyModel(seed) for seed in (0, 1, 2)]


class TrueDrafter(ListDrafter):
    """The true continuation (with its flags), optionally edited per proposal by `edit(atoms, round_index)`."""

    def __init__(self, ref_raw, edit=None):
        super().__init__(None)
        self.ref, self.edit, self.i, self.proposals = ref_raw, edit, 0, []

    def propose(self, k):
        atoms = atoms_of(self.ref[self.i:self.i + k])
        if self.edit is not None:
            atoms = self.edit(atoms)
        self.proposals.append(list(atoms))
        return atoms

    def push(self, atoms):
        self.i += len(atoms)


def flagged_count(raws):
    return sum(1 for r in raws if r >= BOUNDARY)


def test_random_flagged_drafts_are_exact(models):
    rng = random.Random(11)
    n_cross = 0
    for m in models:
        for raw0 in RAW0S:
            ref = [strip(r) for r in plain_decode(m, raw0, 80)]
            for _ in range(6):
                def fn(ctx, k, rng=rng):
                    return [rng.randrange(NB) | (rng.randint(0, 1) << 8) for _ in range(rng.randint(0, k))]

                st = gated.GatedStats()
                got, eng = run_gated(m, raw0, 80, ListDrafter(fn), stats=st)
                assert got == ref and st.emitted == 80
                n_cross += eng.cross_rounds
    assert n_cross > 0


def test_right_drafts_cross_several_patches_and_are_exact(models):
    seen_multi = seen_inside = 0
    for m in models:
        for raw0 in RAW0S:
            ref_raw = plain_decode(m, raw0, 120)
            rounds: list = []
            st = gated.GatedStats()
            got, eng = run_gated(m, raw0, 120, TrueDrafter(ref_raw), stats=st, rounds=rounds)
            assert got == [strip(r) for r in ref_raw]
            emitted_all = []
            for r in rounds[:-1]:  # the last round is cut by the output cap after its inputs were consumed
                emitted_all += r["emitted"]
                consumed = [raw0] + emitted_all[:-1]  # every emitted atom but the last has been an input
                assert r["pos"] == flagged_count(consumed)
                seen_multi += r["crossing"] and r["kept"] >= 2
                seen_inside += any(x >= BOUNDARY for x in r["emitted"][:-1])
            assert st.kinds["cross"] > 0 and st.rounds < 120
    assert seen_multi > 0 and seen_inside > 0, "no crossing round kept two patches; pick other seeds"


def test_a_wrong_flag_ends_acceptance_exactly_at_the_flag(models):
    checked = 0
    for j_bad in (0, 1, 2, 4):
        for m in models:
            for raw0 in RAW0S:
                ref_raw = plain_decode(m, raw0, 100)

                def flip(atoms, j=j_bad):
                    return [a ^ 256 if i == j else a for i, a in enumerate(atoms)]

                d = TrueDrafter(ref_raw, flip)
                rounds: list = []
                got, _ = run_gated(m, raw0, 100, d, rounds=rounds)
                assert got == [strip(r) for r in ref_raw]
                for r, prop in zip(rounds, d.proposals):
                    if r["crossing"] and len(prop) == r["k"] > j_bad:
                        assert r["L"] == j_bad, (j_bad, r)  # d_{j+1} has the wrong flag: only j drafts are accepted
                        checked += 1
    assert checked > 0


def test_a_wrong_byte_after_a_correct_boundary_keeps_exactly_the_patches_before_it(models):
    checked = 0
    for m in models:
        for raw0 in RAW0S:
            ref_raw = plain_decode(m, raw0, 100)
            first_flag = {}

            def edit(atoms):
                idx = next((i for i, a in enumerate(atoms) if a >> 8), None)
                if idx is not None and idx + 1 < len(atoms):
                    atoms[idx + 1] = (((atoms[idx + 1] & 255) + 1) % NB) | (atoms[idx + 1] & 256)
                first_flag[len(first_flag)] = idx
                return atoms

            d = TrueDrafter(ref_raw, edit)
            rounds: list = []
            got, _ = run_gated(m, raw0, 100, d, rounds=rounds)
            assert got == [strip(r) for r in ref_raw]
            for n, (r, prop) in enumerate(zip(rounds, d.proposals)):
                idx = first_flag[n]
                if r["crossing"] and idx is not None and idx + 1 < len(prop) == r["k"] and sum(a >> 8 for a in prop[:idx + 1]) == 1:
                    assert r["L"] == idx + 1 and r["kept"] == 2, (idx, r)  # a0's patch and the one the correct flag opened
                    checked += 1
    assert checked > 0


def test_more_patches_than_slots_are_cut_and_the_output_stays_exact(models):
    checked = 0
    for m in models:
        for raw0 in RAW0S:
            ref_raw = plain_decode(m, raw0, 100)
            for cap_p in (2, 3):
                st = gated.GatedStats()
                got, eng = run_gated(m, raw0, 100, TrueDrafter(ref_raw), max_patches=cap_p, stats=st)
                assert got == [strip(r) for r in ref_raw]
                assert eng.max_p <= cap_p
                checked += st.cap_cut
            # the global cache fills up: capacity just above the patches the whole text needs
            need = flagged_count([raw0] + ref_raw[:99])
            for extra in (0, 1, 2, 5):
                got, eng = run_gated(m, raw0, 100, TrueDrafter(ref_raw), capacity=need + extra)
                assert got == [strip(r) for r in ref_raw]
                assert eng.pos <= need + extra
    assert checked > 0, "the patch cap never cut a draft; pick other seeds"


def test_a_full_global_cache_raises(models):
    m = models[0]
    raw0 = RAW0S[1]
    ref_raw = plain_decode(m, raw0, 60)
    need = flagged_count([raw0] + ref_raw[:59])
    with pytest.raises((RuntimeError, AssertionError)):
        run_gated(m, raw0, 60, TrueDrafter(ref_raw), capacity=need - 1)


def test_the_output_cap_truncates_a_crossing_round(models):
    m = models[0]
    for raw0 in RAW0S:
        ref_raw = plain_decode(m, raw0, 60)
        for n_new in range(1, 30):
            got, _ = run_gated(m, raw0, n_new, TrueDrafter(ref_raw))
            assert got == [strip(r) for r in ref_raw[:n_new]]


def test_eos_inside_a_crossing_round_stops_there(models):
    m = models[1]
    raw0 = RAW0S[1]
    ref_raw = plain_decode(m, raw0, 80)
    for idx in (5, 17, 31):
        target = ref_raw[idx]
        eos_ids = (target,)
        first = ref_raw.index(target)
        got, _ = run_gated(m, raw0, 80, TrueDrafter(ref_raw), eos_ids=eos_ids)
        assert got == [strip(r) for r in ref_raw[:first + 1]]


def test_unavailable_lengths_round_down_and_flags_are_recounted(models):
    m = models[2]
    raw0 = RAW0S[1]
    ref_raw = plain_decode(m, raw0, 90)
    st = gated.GatedStats()
    got, eng = run_gated(m, raw0, 90, TrueDrafter(ref_raw), ks=(0, 3), k_max=3, k_cross=8, stats=st)
    assert got == [strip(r) for r in ref_raw]
    assert set(st.by_k) <= {0, 3}


def test_a_flagger_replaces_the_drafts_own_flags(models):
    m = models[0]
    raw0 = RAW0S[1]
    ref_raw = plain_decode(m, raw0, 90)
    calls = []

    def flagger(ctx, draft, k):
        calls.append(bytes(ctx))
        return atoms_of(ref_raw[len(ctx) - 1: len(ctx) - 1 + len(draft)])  # the true flags, from the context length

    rounds: list = []
    st = gated.GatedStats()
    d = TrueDrafter(ref_raw, lambda atoms: [a & 255 for a in atoms])  # lookup draft without flags
    got, _ = run_gated(m, raw0, 90, d, flagger=flagger, rounds=rounds, stats=st)
    assert got == [strip(r) for r in ref_raw]
    assert calls and st.kinds["cross"] > 0 and st.retok_calls == len(calls)
    assert calls[0] == bytes([strip(raw0) - BYTE_OFFSET])  # the context handed to the flagger starts with a0


def test_plan_crossing_indices():
    flags = [1, 0, 0, 1, 0, 1]
    assert gated.patch_select(flags) == [0, 0, 0, 1, 1, 2]
    assert gated.pool_positions(flags) == [0, 3, 5]
    plan = gated.plan_crossing(BYTE_OFFSET + 2 + BOUNDARY, 0, 1, [5, 6 | 256, 7, 1 | 256], ToyAutomaton(), BYTE_OFFSET, BOUNDARY)
    assert plan.ids == [BYTE_OFFSET + 2, BYTE_OFFSET + 5, BYTE_OFFSET + 6, BYTE_OFFSET + 7, BYTE_OFFSET + 1]
    assert plan.raw_ids == [BYTE_OFFSET + 2, BYTE_OFFSET + 5, BYTE_OFFSET + 6 + BOUNDARY, BYTE_OFFSET + 7,
                            BYTE_OFFSET + 1 + BOUNDARY]
    assert plan.patch_select == [0, 0, 1, 1, 2] and plan.pool == [0, 2, 4] and plan.n_patches == 3
    psel = torch.tensor(plan.patch_select)
    assert [int(gated.patches_kept(psel, torch.tensor(L))) for L in range(5)] == [1, 1, 2, 2, 3]


# ---- injected bugs: each must break the exactness / bookkeeping of some scenario ----------------------------------------


def scenario_fails(models) -> bool:
    """True when a scenario with right drafts diverges, raises, breaks the position bookkeeping or never accepts a flag."""
    kept_two = 0
    for m in models:
        for raw0 in RAW0S:
            ref_raw = plain_decode(m, raw0, 100)
            rounds: list = []
            try:
                got, _ = run_gated(m, raw0, 100, TrueDrafter(ref_raw), max_patches=4, rounds=rounds)
            except (AssertionError, RuntimeError, IndexError):
                return True
            if got != [strip(r) for r in ref_raw]:
                return True
            kept_two += sum(1 for r in rounds if r["crossing"] and r["kept"] >= 2)

            def break_after_flag(atoms):  # a wrong byte right after the first flagged draft atom: a partial acceptance
                idx = next((i for i, a in enumerate(atoms) if a >> 8), None)
                if idx is not None and idx + 1 < len(atoms):
                    atoms[idx + 1] = (((atoms[idx + 1] & 255) + 1) % NB) | (atoms[idx + 1] & 256)
                return atoms

            def flip_flag(atoms):  # right bytes, one wrong flag
                return [a ^ 256 if i == 1 else a for i, a in enumerate(atoms)]

            for edit in (break_after_flag, flip_flag):
                try:
                    got, _ = run_gated(m, raw0, 100, TrueDrafter(ref_raw, edit), max_patches=4)
                except (AssertionError, RuntimeError, IndexError):
                    return True
                if got != [strip(r) for r in ref_raw]:
                    return True
    return kept_two == 0  # right drafts must be accepted across a flag (a bug that only loses acceptance stays exact)


def test_the_scenario_passes_without_a_bug(models):
    assert not scenario_fails(models)


def test_injected_bugs_are_caught(models, monkeypatch):
    def late_select(flags):  # the vector of the previous patch at the flagged position
        out, n = [], 0
        for f in flags:
            out.append(max(n - 1, 0))
            n += int(bool(f))
        return out

    bugs = {
        "patch_select uses the previous patch at a flag": ("patch_select", late_select),
        "pool positions shifted by one": ("pool_positions", lambda flags: [min(j + 1, len(flags) - 1) for j, f in enumerate(flags) if f]),
        "raw draft ids drop the flag": ("raw_draft_ids", lambda draft, bo, bd: [(a & 255) + bo for a in draft]),
        "rollback keeps all patches": ("rollback_index", lambda psel, length: psel[-1:]),
        "accept ignores the flag": ("accepted_raw", lambda raw, tok, s: core.accepted_length_t(raw % BOUNDARY, tok % BOUNDARY, s)),
        "patch select lags one patch behind": ("patch_select", lambda flags: [max(p - 1, 0) for p in gated_patch_select(flags)]),
    }
    for name, (attr, fn) in bugs.items():
        with monkeypatch.context() as mp:
            mp.setattr(gated, attr, fn)
            assert scenario_fails(models), f"bug not caught: {name}"


def gated_patch_select(flags):
    out, n = [], 0
    for f in flags:
        n += int(bool(f))
        out.append(n - 1)
    return out
