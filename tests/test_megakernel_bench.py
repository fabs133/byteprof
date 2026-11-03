"""Tests for the megakernel-theory benchmark kernels (byteprof/bench): streaming outputs against a torch reference,
publication through the counters, and the spin-loop cap (a missing producer must end in the error flag, not a hang).
"""

from __future__ import annotations

import time

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TOL = 1e-4   # observed ~1e-6 (fp32 summation order); a wrong dequant offset or a stale read is >= 1e-2


@pytest.fixture(scope="module")
def sb():
    from byteprof.bench import sync_bench
    return sync_bench


@pytest.fixture(scope="module")
def sbn():
    from byteprof.bench import stream_bench
    return stream_bench


# ---- Stage 1: synchronization ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode,nprod", [(0, 1), (1, 1), (1, 4)])
def test_sync_publishes_data(sb, mode, nprod):
    st = sb.SyncState(40, 8, 128)
    r = sb.run_checked(st, 40, mode, nprod=nprod)
    assert r == {"err": 0, "bad": 0}


def test_sync_launch_twice_without_reset_uses_base(sb):
    st = sb.SyncState(10, 4, 128)
    sb.launch_chain(st, 10, 0)
    torch.cuda.synchronize()
    st.base += 10 * st.grid
    sb.launch_chain(st, 10, 0)
    torch.cuda.synchronize()
    assert int(st.err.item()) == 0 and int(st.bad.item()) == 0


@pytest.mark.parametrize("mode,kw", [(0, dict(skip_block=1, skip_phase=2)), (1, dict(skip_block=1, skip_phase=2, nprod=2))])
def test_sync_missing_producer_ends_in_error_flag(sb, mode, kw):
    st = sb.SyncState(10, 4, 128)
    t0 = time.time()
    r = sb.run_checked(st, 10, mode, cap_ns=2_000_000, **kw)
    assert r["err"] == 1
    assert time.time() - t0 < 1.0   # far below the ~2 s display-driver watchdog


def test_sync_grid_larger_than_resident_is_refused(sb):
    big = sb.max_coresident(256) + 1
    st = sb.SyncState(4, big, 256)
    with pytest.raises(AssertionError):
        sb.launch_chain(st, 4, 0)


def test_slope_fit(sb):
    b, a = sb.slope_us_per_phase([10, 20, 40, 80], [15, 25, 45, 85])
    assert abs(b - 1.0) < 1e-9 and abs(a - 5.0) < 1e-9


# ---- Stage 2: streaming kernels ----------------------------------------------------------------------------------------
def test_row_stride(sbn):
    assert sbn.row_stride("bf16", 2048) == 4096
    assert sbn.row_stride("int8", 2816) == 2816
    assert sbn.row_stride("int4", 2048) == 1056 and sbn.row_stride("int4", 2816) % 16 == 0


@pytest.mark.parametrize("q", ["bf16", "int8", "int4"])
def test_quantize_round_trip_error_is_half_a_step(sbn, q):
    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    w = torch.randn(64, 2816, device="cuda", generator=g)
    raw, rs, deq = sbn.quantize(w, q)
    assert raw.shape == (64, sbn.row_stride(q, 2816))
    err = (deq - w).abs()
    if q == "bf16":
        assert float((err / w.abs().clamp_min(1e-3)).max()) < 0.01
    elif q == "int8":
        assert bool((err <= rs[:, None] * 0.5001).all())
    else:
        step = (w.view(64, -1, 128).abs().amax(-1) / 7).clamp_min(1e-6).half().float().repeat_interleave(128, dim=1)
        assert bool((err <= step * 0.5001 + 1e-6).all())


@pytest.fixture(scope="module", params=["bf16", "int8", "int4"])
def stack(request, sbn):
    st = sbn.Stack(request.param, keep_deq=True, seed=3)
    st.ref1 = st.reference(1)
    st.ref2 = st.reference(2)
    yield st
    del st
    torch.cuda.empty_cache()


def _fresh(st):
    st.reset_state()
    st.buf["x"].zero_()
    st.buf["logits"].zero_()


@pytest.mark.parametrize("prefetch", [0, 1, 2])
def test_persistent_matches_reference(sbn, stack, prefetch):
    _fresh(stack)
    stack.mega(prefetch)
    torch.cuda.synchronize()
    assert int(stack.ctl[2].item()) == 0
    errs = sbn.compare(stack.fresh_outputs(1), stack.ref1)
    assert max(errs.values()) < TOL, errs


@pytest.mark.parametrize("prefetch", [0, 2])
def test_persistent_two_steps_matches_reference(sbn, stack, prefetch):
    """The second launch uses the swapped n/m buffers and the advanced counter base."""
    _fresh(stack)
    stack.mega(prefetch)
    stack.mega(prefetch)
    torch.cuda.synchronize()
    assert int(stack.ctl[2].item()) == 0
    errs = sbn.compare(stack.fresh_outputs(2), stack.ref2)
    assert max(errs.values()) < TOL, errs


@pytest.mark.parametrize("grid", [38, 152])
def test_per_phase_kernels_match_reference(sbn, stack, grid):
    _fresh(stack)
    stack.per_phase(grid)
    torch.cuda.synchronize()
    errs = sbn.compare(stack.fresh_outputs(1), stack.ref1)
    assert max(errs.values()) < TOL, errs


def test_persistent_graph_replays_match_reference(sbn, stack):
    """Captured in a CUDA graph and replayed twice: same as two direct launches."""
    stack.sw_bytes = sbn.SW_BYTES
    _fresh(stack)
    g = stack.graph_of(lambda: stack.mega(1))
    _fresh(stack)
    g.replay()
    g.replay()
    torch.cuda.synchronize()
    errs = sbn.compare(stack.fresh_outputs(2), stack.ref2)
    assert max(errs.values()) < TOL, errs


def test_persistent_missing_producer_ends_in_error_flag(stack):
    _fresh(stack)
    t0 = time.time()
    stack.mega(1, cap_ns=3_000_000, skip_block=5, skip_phase=3)
    torch.cuda.synchronize()
    assert int(stack.ctl[2].item()) == 1
    assert time.time() - t0 < 1.0
    _fresh(stack)   # leaves the counters clean for the next test
    stack.mega(1)
    torch.cuda.synchronize()
    assert int(stack.ctl[2].item()) == 0


def test_persistent_refuses_non_resident_grid(stack):
    with pytest.raises(AssertionError):
        stack.mega(0, grid=38 * 3)   # one 94 KiB block per SM fits; three per SM cannot be co-resident


def test_wrong_weights_are_detected(sbn, stack):
    """The comparison has teeth: corrupt one layer's weights and the error must exceed the tolerance."""
    _fresh(stack)
    w = stack.layers[2]["wgate"].raw
    saved = w[:8].clone()
    w[:8] = (w[:8].to(torch.int32) ^ 0x55).to(torch.uint8)
    stack.mega(0)
    torch.cuda.synchronize()
    errs = sbn.compare(stack.fresh_outputs(1), stack.ref1)
    w[:8] = saved
    assert max(errs.values()) > 10 * TOL
