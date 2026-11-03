"""GPU checks of the fused verification kernels (TASK_parallel_verify.md, Stage A1): the multi-step kernel and the
lazy commit against T sequential calls of the library's step kernel, for T = 1..33 and every L, fp32 and bf16 inputs;
the acceptance kernel against the reference; the fused path inside a CUDA graph with a dynamic L."""

import pytest
import torch

from byteprof.optim import draft_core as core

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
NH, DV = 16, 128
DQS = (64, 128)  # Bolmo's local layers have head_dim_qk = 64 (one block); 128 exercises the two-block path


def rand_inputs(s, dtype, seed=0, DQ=64):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *shape, dt=dtype: torch.randn(*shape, generator=g, device="cuda", dtype=dt)  # noqa: E731
    return (r(s, NH, DQ), r(s, NH, DQ), r(s, NH, DV), r(s, NH, 1) * 2, r(s, NH, 1) * 2,
            (r(1, NH, DQ, DV, dt=torch.float32) * 0.1, r(1, NH, DQ, dt=torch.float32) * 0.1,
             r(1, NH, 1, dt=torch.float32)))


def reference(state, q, k, v, i, f):
    """T sequential library step calls: the outputs and the state after every position."""
    from mlstm_kernels.torch.recurrent.triton_step import mlstm_recurrent_step__triton_fw

    hs, states, prev = [], [], state
    for j in range(q.shape[0]):
        h, new = mlstm_recurrent_step__triton_fw(prev[0], prev[1], prev[2], q[j:j + 1], k[j:j + 1], v[j:j + 1],
                                                 i[j:j + 1], f[j:j + 1], eps=1e-6)
        hs.append(h)
        states.append(new)
        prev = new
    return torch.cat(hs, 0), states


def rel(a, b):
    """Max abs error relative to the largest magnitude of the reference (a normwise relative error)."""
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-30))


@cuda
@pytest.mark.parametrize("DQ", DQS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_multistep_outputs_and_lazy_commit_match_the_step_kernel_for_every_t_and_l(dtype, DQ):
    from byteprof.optim import draft_fused as fused

    tol = 1e-5 if dtype == torch.float32 else 1e-2
    worst_h = worst_s = 0.0
    for t in (1, 2, 5, 9, 17, 33):
        q, k, v, i, f, state = rand_inputs(t, dtype, seed=t, DQ=DQ)
        h_ref, states = reference(state, q, k, v, i, f)
        cur = tuple(x.clone() for x in state)
        hbuf = torch.zeros(40, NH, DV, dtype=dtype, device="cuda")
        ns, ms = torch.zeros(NH, DQ, device="cuda"), torch.zeros(NH, device="cuda")
        fused.multistep_fw(cur, q, k, v, i, f, hbuf, ns, ms, eps=1e-6)
        assert all(torch.equal(a, b) for a, b in zip(cur, state))  # the pass never writes the state
        err = rel(hbuf[:t], h_ref)
        worst_h = max(worst_h, err)
        assert err < tol, (t, err)
        for length in range(t):
            cur2 = tuple(x.clone() for x in state)
            fused.lazy_commit(cur2, k, v, i, f, torch.tensor(length, device="cuda"), ns, ms)
            for a, b in zip(cur2, states[length]):
                e = rel(a, b)
                worst_s = max(worst_s, e)
                assert e < tol, (t, length, e)
    print("max relative error", dtype, "DQ", DQ, "h:", worst_h, "state:", worst_s)


@cuda
def test_fp32_outputs_agree_to_fp32_rounding():
    from byteprof.optim import draft_fused as fused

    DQ = 64
    q, k, v, i, f, state = rand_inputs(9, torch.float32)
    h_ref, states = reference(state, q, k, v, i, f)
    hbuf = torch.zeros(9, NH, DV, device="cuda")
    ns, ms = torch.zeros(NH, DQ, device="cuda"), torch.zeros(NH, device="cuda")
    fused.multistep_fw(state, q, k, v, i, f, hbuf, ns, ms, eps=1e-6)
    assert rel(hbuf, h_ref) < 5e-6  # fp32 rounding only (FMA / transcendental differences)


@cuda
def test_lazy_commit_in_a_graph_with_a_dynamic_length_and_in_place_state():
    from byteprof.optim import draft_fused as fused

    DQ = 64
    t = 7
    q, k, v, i, f, state = rand_inputs(t, torch.bfloat16, seed=3)
    _, states = reference(state, q, k, v, i, f)
    cur = tuple(x.clone() for x in state)
    start = tuple(x.clone() for x in state)
    hbuf = torch.zeros(8, NH, DV, dtype=torch.bfloat16, device="cuda")
    ns, ms = torch.zeros(NH, DQ, device="cuda"), torch.zeros(NH, device="cuda")
    length = torch.zeros((), dtype=torch.long, device="cuda")
    fused.multistep_fw(cur, q, k, v, i, f, hbuf, ns, ms, eps=1e-6)  # warm up (compile)
    fused.lazy_commit(cur, k, v, i, f, length, ns, ms)
    for a, b in zip(cur, start):
        a.copy_(b)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fused.multistep_fw(cur, q, k, v, i, f, hbuf, ns, ms, eps=1e-6)
        fused.lazy_commit(cur, k, v, i, f, length, ns, ms)
    for length_v in (0, 3, t - 1, 1):
        for a, b in zip(cur, start):
            a.copy_(b)
        length.fill_(length_v)
        g.replay()
        torch.cuda.synchronize()
        for a, b in zip(cur, states[length_v]):
            assert rel(a, b) < 1e-2, (length_v, rel(a, b))


@cuda
def test_accept_kernel_equals_the_reference():
    from byteprof.optim import draft_fused as fused

    ids = torch.tensor([10, 5, 6, 7, 8, 9], device="cuda")
    for tokl, want in (([5, 6, 99, 8, 9, 3], 2), ([5, 6, 7, 8, 9 + 260, 3], 4), ([1, 6, 7, 8, 9, 3], 0),
                       ([5, 6, 7, 8, 9, 3], 5)):
        tok = torch.tensor(tokl, device="cuda")
        out = torch.zeros(8, dtype=torch.long, device="cuda")
        fused.accept(ids, tok, out, 6, 6)
        assert int(out[6]) == want == int(core.accepted_length_t(ids, tok, 6))
        assert out[:6].tolist() == tokl
