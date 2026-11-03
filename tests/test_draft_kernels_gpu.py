"""GPU checks of the drafting kernels: the step-into wrapper is the library's step kernel (bit-identical), the scan
writes the right history slots, the commit kernel selects the slot on the device, also inside a CUDA graph."""

from functools import partial

import pytest
import torch

from byteprof.optim import draft_core as core

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
NH, DQ, DV = 16, 128, 128


def rand_inputs(s, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *shape, dt=dtype: torch.randn(*shape, generator=g, device="cuda", dtype=dt)  # noqa: E731
    return (r(s, NH, DQ), r(s, NH, DQ), r(s, NH, DV), r(s, NH, 1), r(s, NH, 1),
            (r(1, NH, DQ, DV, dt=torch.float32) * 0.1, r(1, NH, DQ, dt=torch.float32) * 0.1,
             r(1, NH, 1, dt=torch.float32)))


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_step_into_equals_library_step_and_scan_fills_the_history(dtype):
    from mlstm_kernels.torch.recurrent.triton_step import mlstm_recurrent_step__triton_fw

    s = 5
    q, k, v, i, f, state = rand_inputs(s, dtype)
    hist = (torch.zeros(9, 1, NH, DQ, DV, device="cuda"), torch.zeros(9, 1, NH, DQ, device="cuda"),
            torch.zeros(9, 1, NH, 1, device="cuda"))
    hbuf = torch.zeros(9, NH, DV, dtype=dtype, device="cuda")
    cur = tuple(t.clone() for t in state)
    core.scan_with_history(partial(core.mlstm_step_into, eps=1e-6), cur, hist, s, q, k, v, i, f, hbuf)
    assert all(torch.equal(a, b) for a, b in zip(cur, state))  # the current state is not touched by the scan
    prev = state
    for j in range(s):
        h_ref, new = mlstm_recurrent_step__triton_fw(
            prev[0], prev[1], prev[2], q[j:j + 1], k[j:j + 1], v[j:j + 1], i[j:j + 1], f[j:j + 1], eps=1e-6)
        assert torch.equal(h_ref, hbuf[j:j + 1])
        for a, hh in zip(new, hist):
            assert torch.equal(a, hh[j])
        prev = new


@cuda
def test_commit_selects_the_slot_from_the_device_value_also_in_a_graph():
    s = 6
    hist = (torch.randn(9, 1, NH, DQ, DV, device="cuda"), torch.randn(9, 1, NH, DQ, device="cuda"),
            torch.randn(9, 1, NH, 1, device="cuda"))
    cur = tuple(torch.zeros_like(h[0]) for h in hist)
    length = torch.zeros((), dtype=torch.long, device="cuda")
    core.commit_state(hist, cur, length)  # warm up (compiles the kernel)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        core.commit_state(hist, cur, length)
    for slot in (0, 3, s, 8, 1):
        length.fill_(slot)
        g.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(c, h[slot]) for c, h in zip(cur, hist))


@cuda
def test_accepted_length_on_the_device():
    ids = torch.tensor([10, 5, 6, 7, 8, 9], device="cuda")
    tok = torch.tensor([5, 6, 99, 8, 9, 3], device="cuda")
    assert int(core.accepted_length_t(ids, tok, 6)) == 2  # d_3 = 7 != 99
    tok2 = torch.tensor([5, 6, 7, 8, 9 + 260, 3], device="cuda")  # a boundary-flagged p_5 is never a plain byte id
    assert int(core.accepted_length_t(ids, tok2, 6)) == 4
