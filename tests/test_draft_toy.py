"""Toy-model exactness test of the drafting loop (CPU, fp32, nothing to download).

A tiny random-weight latent-tokenizer model - fused (byte, boundary-flag) atoms, an encoder with a recurrent local
layer fed by a byte embedding plus a subword id from a stateful automaton, patches (the encoder output at a flagged
byte goes through a recurrent "global" layer), and a decoder with a recurrent local layer and an lm_head over fused
atoms - decoded (a) one atom at a time ("plain") and (b) by the drafting loop with random / all-wrong / all-right
drafts. The drafting engine uses the production code: `scan_with_history`, `commit_state`, `accepted_length_t`
and `rounds_loop` of byteprof.optim.draft_core. This is the test that catches rollback and indexing bugs.
"""

import random

import pytest
import torch
import torch.nn as nn

from byteprof.optim import draft_core as core

BYTE_OFFSET, BOUNDARY = 4, 260
NB = 8  # byte values 0..7
NSUB = 5
D, NH, DQ, DV = 16, 2, 4, 8
VOCAB = 2 * BOUNDARY
EOS = 2
EOS_IDS = (EOS, EOS + BOUNDARY)
BPE_END = 3


def toy_step(prev, new, q, k, v, i, f, h):
    """One step of a small matrix-memory recurrence; writes `new` and `h` in place (prev is not modified)."""
    c_old, n_old, m_old = prev
    ig, fg = torch.sigmoid(i), torch.sigmoid(f)  # (1,NH,1)
    new[0].copy_(fg.unsqueeze(-1) * c_old + ig.unsqueeze(-1) * k.unsqueeze(-1) * v.unsqueeze(-2))
    new[1].copy_(fg * n_old + ig * k)
    new[2].copy_(fg * m_old + ig)
    num = torch.einsum("bhq,bhqv->bhv", q, new[0])
    den = (q * new[1]).sum(-1, keepdim=True).abs().clamp(min=1.0) + new[2].abs()
    h.copy_(num / den)


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D, elementwise_affine=False)
        self.q = nn.Linear(D, NH * DQ)
        self.k = nn.Linear(D, NH * DQ)
        self.v = nn.Linear(D, NH * DV)
        self.i = nn.Linear(D, NH)
        self.f = nn.Linear(D, NH)
        self.o = nn.Linear(NH * DV, D)
        self.mlp = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        with torch.no_grad():  # long memory and a large state contribution: a wrong state must change the output
            self.f.bias.fill_(2.0)
            self.o.weight.mul_(4.0)
            for lin in (self.q, self.k, self.v):
                lin.weight.mul_(2.0)

    def new_state(self):
        return (torch.zeros(1, NH, DQ, DV), torch.zeros(1, NH, DQ), torch.zeros(1, NH, 1))

    def new_hist(self, smax):
        return (torch.zeros(smax, 1, NH, DQ, DV), torch.zeros(smax, 1, NH, DQ), torch.zeros(smax, 1, NH, 1))

    def plain(self, h, state):
        """One position, state updated in place (what the minimal loop does)."""
        x = self.norm(h)
        tmp = tuple(torch.empty_like(t) for t in state)
        hb = torch.empty(1, NH, DV)
        toy_step(state, tmp, self.q(x).view(1, NH, DQ), self.k(x).view(1, NH, DQ), self.v(x).view(1, NH, DV),
                 self.i(x).view(1, NH, 1), self.f(x).view(1, NH, 1), hb)
        for dst, src in zip(state, tmp):
            dst.copy_(src)
        h = h + self.o(hb.reshape(1, 1, -1))
        return h + self.mlp(h)

    def multi(self, h, s, cur, hist, hbuf, saved=None):
        """S positions: batched projections, S sequential steps writing the history (`saved is None`), or, for the lazy
        commit, only the cell outputs; `saved` then receives (start state, q, k, v, i, f) for `lazy_commit_state`."""
        x = self.norm(h)
        q, k, v = self.q(x).view(s, NH, DQ), self.k(x).view(s, NH, DQ), self.v(x).view(s, NH, DV)
        i, f = self.i(x).view(s, NH, 1), self.f(x).view(s, NH, 1)
        if saved is None:
            core.scan_with_history(toy_step, cur, hist, s, q, k, v, i, f, hbuf)
        else:
            saved[:] = [core.scan_without_history(toy_step, cur, s, q, k, v, i, f, hbuf), q, k, v, i, f]
        h = h + self.o(hbuf[:s].reshape(1, s, -1))
        return h + self.mlp(h)


class ToyModel(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.byte_emb = nn.Embedding(VOCAB, D)
        self.sub_emb = nn.Embedding(NSUB, D)
        self.enc = ToyLayer()
        self.out_proj = nn.Linear(D, D)
        self.glob_w = nn.Linear(2 * D, D)
        self.in_proj = nn.Linear(D, D)
        self.dec = ToyLayer()
        self.norm = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, VOCAB)
        with torch.no_grad():
            valid = [b + BYTE_OFFSET + fl * BOUNDARY for b in range(NB) for fl in (0, 1)]
            bias = torch.full((VOCAB,), -1e4)
            bias[valid] = 0.0
            self.lm_head.bias.copy_(bias)
            self.lm_head.weight.mul_(4.0)
        self.eval()
        self.reset()

    def reset(self):
        self.enc_state, self.dec_state = self.enc.new_state(), self.dec.new_state()
        self.g = torch.zeros(1, D)
        self.last_value = torch.zeros(1, 1, D)

    def global_update(self, patch):  # recurrent "global model": only runs on flagged positions
        self.g = torch.tanh(self.glob_w(torch.cat([self.g, patch.reshape(1, D)], dim=-1)))
        self.last_value = self.g.view(1, 1, D).clone()

    def plain_step(self, byte_id, sub_id, boundary):
        h = self.byte_emb(torch.tensor([[byte_id]])) + self.sub_emb(torch.tensor([[sub_id]]))
        h = self.enc.plain(h, self.enc_state)
        if boundary:
            self.global_update(self.out_proj(h))
        h = self.dec.plain(self.in_proj(h) + self.last_value, self.dec_state)
        return int(torch.argmax(self.lm_head(self.norm(h)), dim=-1)[0, -1])


class ToyAutomaton:
    """Stateful subword id: depends on the previous byte (state), so a wrong state rollback changes outputs."""

    def step(self, state, byte):
        if byte >= BOUNDARY:
            byte -= BOUNDARY
        return byte, (byte * 7 + state * 3) % NSUB


class ToyEngine:
    """RoundEngine over the toy model, built on the production scan/accept/commit code."""

    def __init__(self, model: ToyModel, ks=(0, 1, 2, 3, 5, 8), verify="steps"):
        self.m = model
        self.verify = verify
        self.k_values = tuple(ks)
        smax = max(ks) + 1
        self.enc_hist, self.dec_hist = model.enc.new_hist(smax), model.dec.new_hist(smax)
        self.enc_buf, self.dec_buf = torch.zeros(smax, NH, DV), torch.zeros(smax, NH, DV)

    def run_round(self, k, boundary, ids, subs):
        m, s = self.m, k + 1
        ids_t = torch.tensor(ids)
        h = m.byte_emb(ids_t.view(1, s)) + m.sub_emb(torch.tensor(subs).view(1, s))
        lazy = self.verify == "lazy"
        enc_saved, dec_saved = ([], []) if lazy else (None, None)
        h = m.enc.multi(h, s, m.enc_state, self.enc_hist, self.enc_buf, enc_saved)
        if boundary:
            m.global_update(m.out_proj(h[:, :1]))
        h = m.dec.multi(m.in_proj(h) + m.last_value, s, m.dec_state, self.dec_hist, self.dec_buf, dec_saved)
        tok = torch.argmax(m.lm_head(m.norm(h)), dim=-1)[0]
        length = core.accepted_length_t(ids_t, tok, s)
        if lazy:
            core.lazy_commit_state(toy_step, m.enc_state, enc_saved[0], length, *enc_saved[1:])
            core.lazy_commit_state(toy_step, m.dec_state, dec_saved[0], length, *dec_saved[1:])
        else:
            core.commit_state(self.enc_hist, m.enc_state, length)
            core.commit_state(self.dec_hist, m.dec_state, length)
        return tok.tolist(), int(length)


def plain_decode(model, raw0, n_new, eos_ids=EOS_IDS):
    model.reset()
    aut = ToyAutomaton()
    state, sub = aut.step(0, raw0)
    raw, out = raw0, []
    for _ in range(n_new):
        boundary = raw >= BOUNDARY
        a0_id = raw - BOUNDARY if boundary else raw
        raw = model.plain_step(a0_id, sub, boundary)
        out.append(raw)
        if raw in eos_ids:
            break
        state, sub = aut.step(state, raw)
    return out


class ListDrafter:
    """Proposes from a function of (context, k): random, wrong, right or mixed drafts."""

    def __init__(self, fn):
        self.fn, self.ctx = fn, []

    def reset(self, ctx_atoms, truth=None):
        self.ctx = list(ctx_atoms)

    def propose(self, k):
        return self.fn(self.ctx, k)

    def push(self, atoms):
        self.ctx.extend(atoms)


def atoms_of(raws):
    return [core.atom_of(r, BYTE_OFFSET, BOUNDARY) for r in raws]


def run_draft(model, raw0, n_new, drafter, ks=(0, 1, 2, 3, 5, 8), k_max=8, eos_ids=EOS_IDS, stats=None, rounds=None):
    model.reset()
    eng = ToyEngine(model, ks, verify=VERIFY[0])
    aut = ToyAutomaton()
    state, sub = aut.step(0, raw0)
    stop_len = 1 + n_new
    tokens = [0] * (stop_len + 2)
    n_tok = 1
    drafter.reset([core.atom_of(raw0, BYTE_OFFSET, BOUNDARY)], None)
    n_tok = core.rounds_loop(eng, aut, drafter, state=state, sub=sub, raw=raw0, tokens=tokens, n_tok=n_tok,
                             stop_len=stop_len, eos_ids=eos_ids, bpe_end=BPE_OFFSET_NONE, byte_offset=BYTE_OFFSET,
                             boundary_offset=BOUNDARY, k_max=k_max, stats=stats,
                             on_round=(rounds.append if rounds is not None else None))
    return tokens[1:n_tok]


BPE_OFFSET_NONE = -1  # the toy never produces <bpe_token_end>
VERIFY = ["steps"]  # the verification path under test; the autouse fixture below runs every test on both


@pytest.fixture(params=["steps", "lazy"], autouse=True)
def verify_path(request):
    VERIFY[0] = request.param
    yield request.param
    VERIFY[0] = "steps"


def strip(r):
    return r - BOUNDARY if r >= BOUNDARY else r


def reference(model, raw0, n_new, eos_ids=EOS_IDS):
    return [strip(r) for r in plain_decode(model, raw0, n_new, eos_ids)]


@pytest.fixture(scope="module")
def models():
    return [ToyModel(seed) for seed in (0, 1, 2)]


RAW0S = (4 + 3, 4 + 5 + BOUNDARY)  # a0 unflagged and flagged


@pytest.mark.parametrize("n_new", [1, 2, 7, 40, 64])
def test_k0_only_loop_equals_plain_decoding(models, n_new):
    for m in models:
        for raw0 in RAW0S:
            got = run_draft(m, raw0, n_new, ListDrafter(lambda c, k: []), ks=(0,))
            assert got == reference(m, raw0, n_new)


def test_random_drafts_are_exact(models):
    rng = random.Random(7)
    for m in models:
        for raw0 in RAW0S:
            ref_raw = plain_decode(m, raw0, 80)
            ref = [strip(r) for r in ref_raw]
            for trial in range(6):
                def fn(ctx, k, rng=rng):
                    n = rng.randint(0, k)
                    return [rng.randrange(NB) | (rng.randint(0, 1) << 8) for _ in range(n)]

                st = core.Stats()
                got = run_draft(m, raw0, 80, ListDrafter(fn), stats=st)
                assert got == ref
                assert st.emitted == 80


def test_all_wrong_drafts_accept_nothing_and_are_exact(models):
    for m in models:
        for raw0 in RAW0S:
            ref_raw = plain_decode(m, raw0, 60)
            pos = {"i": 0}

            def fn(ctx, k):
                i = pos["i"]
                return [((ref_raw[i + j] % BOUNDARY - BYTE_OFFSET) + 1) % NB for j in range(min(k, len(ref_raw) - i - 1))]

            class D(ListDrafter):
                def push(self, atoms):
                    pos["i"] += len(atoms)

            st = core.Stats()
            got = run_draft(m, raw0, 60, D(fn), stats=st)
            assert got == [strip(r) for r in ref_raw]
            assert set(st.l_hist) == {0} and st.rounds == 60


def test_all_right_drafts_accept_up_to_the_first_boundary_and_are_exact(models):
    for m in models:
        for raw0 in RAW0S:
            ref_raw = plain_decode(m, raw0, 90)
            pos = {"i": 0}

            def fn(ctx, k):
                i = pos["i"]
                return atoms_of(ref_raw[i:i + k])

            class D(ListDrafter):
                def push(self, atoms):
                    pos["i"] += len(atoms)

            rounds: list = []
            st = core.Stats()
            got = run_draft(m, raw0, 90, D(fn), stats=st, rounds=rounds)
            assert got == [strip(r) for r in ref_raw]
            assert max(st.l_hist) > 0 and st.rounds < 90
            # boundaries inside the draft: a flagged atom ends the round (it is emitted but nothing after it)
            inside = [r for r in rounds if r["k"] > 0 and any(x >= BOUNDARY for x in r["emitted"][:-1])]
            assert not inside
            ended_by_flag = [r for r in rounds if r["k"] > 0 and r["L"] < r["k"] and r["emitted"][-1] >= BOUNDARY]
            assert ended_by_flag, "no round ended at a boundary inside its draft; pick other seeds"


def test_the_cap_truncates_mid_round(models):
    m = models[0]
    raw0 = RAW0S[0]
    ref_raw = plain_decode(m, raw0, 50)
    for n_new in range(1, 30):
        pos = {"i": 0}

        def fn(ctx, k):
            return atoms_of(ref_raw[pos["i"]:pos["i"] + k])

        class D(ListDrafter):
            def push(self, atoms):
                pos["i"] += len(atoms)

        got = run_draft(m, raw0, n_new, D(fn))
        assert got == [strip(r) for r in ref_raw[:n_new]]


def test_stops_at_eos_inside_a_round(models):
    m = models[1]
    raw0 = RAW0S[0]
    ref_raw = plain_decode(m, raw0, 60)
    target = ref_raw[17]  # pretend this id is the EOS: plain and drafting must both stop right after its first use
    eos_ids = (target,)
    ref = reference(m, raw0, 60, eos_ids)
    first = ref_raw.index(target)
    assert len(ref) == first + 1
    pos = {"i": 0}

    def fn(ctx, k):
        return atoms_of(ref_raw[pos["i"]:pos["i"] + k])

    class D(ListDrafter):
        def push(self, atoms):
            pos["i"] += len(atoms)

    assert run_draft(m, raw0, 60, D(fn), eos_ids=eos_ids) == ref


def test_unavailable_draft_lengths_round_down_to_available_graphs(models):
    m = models[2]
    raw0 = RAW0S[1]
    ref_raw = plain_decode(m, raw0, 70)
    pos = {"i": 0}

    def fn(ctx, k):
        return atoms_of(ref_raw[pos["i"]:pos["i"] + 8])

    class D(ListDrafter):
        def push(self, atoms):
            pos["i"] += len(atoms)

    st = core.Stats()
    got = run_draft(m, raw0, 70, D(fn), ks=(0, 3), stats=st)
    assert got == [strip(r) for r in ref_raw]
    assert set(st.by_k) <= {0, 3}


def test_scan_with_history_equals_sequential_steps_and_commit_selects_the_slot():
    torch.manual_seed(3)
    layer = ToyLayer()
    s = 6
    q, k_, v = torch.randn(s, NH, DQ), torch.randn(s, NH, DQ), torch.randn(s, NH, DV)
    i, f = torch.randn(s, NH, 1), torch.randn(s, NH, 1)
    cur = tuple(torch.randn_like(t) for t in layer.new_state())
    hist, hbuf = layer.new_hist(8), torch.zeros(8, NH, DV)
    cur0 = tuple(t.clone() for t in cur)
    core.scan_with_history(toy_step, cur, hist, s, q, k_, v, i, f, hbuf)
    assert all(torch.equal(a, b) for a, b in zip(cur, cur0))  # the current state is untouched until commit
    prev = cur0
    for j in range(s):
        new = layer.new_state()
        h = torch.empty(1, NH, DV)
        toy_step(prev, new, q[j:j + 1], k_[j:j + 1], v[j:j + 1], i[j:j + 1], f[j:j + 1], h)
        assert torch.equal(h, hbuf[j:j + 1])
        assert all(torch.equal(a, hh[j]) for a, hh in zip(new, hist))
        prev = new
    for slot in (0, 3, 5):
        core.commit_state(hist, cur, torch.tensor(slot))
        assert all(torch.equal(c, hh[slot]) for c, hh in zip(cur, hist))


def test_accepted_length_host_and_device_agree():
    rng = random.Random(1)
    for _ in range(300):
        k = rng.randint(0, 8)
        ids = [rng.randrange(4, 12) for _ in range(k + 1)]
        toks = [ids[j + 1] if rng.random() < 0.7 else rng.choice([rng.randrange(4, 12), rng.randrange(264, 272)])
                for j in range(k)] + [rng.randrange(4, 12)]
        host = core.accepted_length(ids, toks, k)
        dev = int(core.accepted_length_t(torch.tensor(ids), torch.tensor(toks), k + 1)) if k > 0 else 0
        assert host == dev


def test_choose_k():
    ks = (0, 1, 2, 4, 6, 8)
    assert core.choose_k(0, ks, 8) == 0
    assert core.choose_k(3, ks, 8) == 2
    assert core.choose_k(8, ks, 8) == 8
    assert core.choose_k(8, ks, 4) == 4
    assert core.choose_k(7, ks, 8) == 6


def test_lazy_commit_state_equals_the_history_slots_and_has_teeth():
    torch.manual_seed(5)
    layer = ToyLayer()
    s = 7
    q, k_, v = torch.randn(s, NH, DQ), torch.randn(s, NH, DQ), torch.randn(s, NH, DV)
    i, f = torch.randn(s, NH, 1), torch.randn(s, NH, 1)
    cur = tuple(torch.randn_like(t) for t in layer.new_state())
    hist, hbuf = layer.new_hist(8), torch.zeros(8, NH, DV)
    core.scan_with_history(toy_step, cur, hist, s, q, k_, v, i, f, hbuf)
    hbuf2 = torch.zeros(8, NH, DV)
    cur2 = tuple(t.clone() for t in cur)
    start = core.scan_without_history(toy_step, cur2, s, q, k_, v, i, f, hbuf2)
    assert torch.equal(hbuf, hbuf2) and all(torch.equal(a, b) for a, b in zip(cur, cur2))
    for slot in range(s):
        out = tuple(t.clone() for t in cur)
        core.lazy_commit_state(toy_step, out, start, torch.tensor(slot), q, k_, v, i, f)
        assert all(torch.equal(a, hh[slot]) for a, hh in zip(out, hist)), slot
    # teeth: the three plausible bugs of a lazy commit all change the state
    out = tuple(t.clone() for t in cur)
    core.lazy_commit_state(toy_step, out, start, torch.tensor(2), q, k_, v, i, f)
    for wrong in (1, 3):  # one position too few / too many
        bad = tuple(t.clone() for t in cur)
        core.lazy_commit_state(toy_step, bad, start, torch.tensor(wrong), q, k_, v, i, f)
        assert not all(torch.equal(a, b) for a, b in zip(bad, out))
    bad = tuple(t.clone() for t in cur)  # recomputing from the already advanced state instead of the start state
    core.lazy_commit_state(toy_step, bad, tuple(t[s - 1] for t in hist), torch.tensor(2), q, k_, v, i, f)
    assert not all(torch.equal(a, b) for a, b in zip(bad, out))
