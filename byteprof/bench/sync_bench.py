"""Stage 1 of TASK_megakernel_theory.md: cost of a synchronized phase on a persistent kernel (one block per SM).

All launches use at most as many blocks as can be co-resident (checked with the occupancy API) and every spin loop
has a wall-clock cap, so a missing producer ends in the error flag instead of a hang (TDR).
"""

from __future__ import annotations

import statistics

import torch

from . import cuda_rt
from .sync_src import SOURCE

_MOD = None
CAP_NS = 50_000_000  # 50 ms per wait; a failed launch ends in well under 0.1 s


def module() -> cuda_rt.Module:
    global _MOD
    if _MOD is None:
        _MOD = cuda_rt.Module(SOURCE)
    return _MOD


def sm_count() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def max_coresident(threads: int, smem: int = 0, name: str = "sync_chain") -> int:
    """Largest grid whose blocks are all resident at once."""
    f = module().func(name)
    return cuda_rt.occupancy(f, threads, smem) * sm_count()


class SyncState:
    def __init__(self, nphase_max: int, grid: int, threads: int):
        self.grid, self.threads = grid, threads
        self.counters = torch.zeros(64 + 1024, dtype=torch.int32, device="cuda")
        self.vals = torch.zeros(nphase_max * grid * threads, dtype=torch.float32, device="cuda")
        self.err = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.bad = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.base = 0

    def reset(self) -> None:
        self.counters.zero_()
        self.err.zero_()
        self.bad.zero_()
        self.base = 0


def launch_chain(st: SyncState, nphase: int, mode: int, nprod: int = 1, work: int = 0, skip_block: int = -1,
                 skip_phase: int = -1, skip_wait_block: int = -1, cap_ns: int = CAP_NS, impl: int = 0) -> None:
    assert st.grid <= max_coresident(st.threads), "grid would not be co-resident"
    f = module().func("sync_chain")
    import ctypes as C
    cuda_rt.launch(f, st.grid, st.threads, [st.counters, st.vals, st.err, st.bad, nphase, mode, nprod, work,
                                            C.c_uint32(st.base & 0xFFFFFFFF), C.c_uint64(cap_ns), skip_block,
                                            skip_phase, skip_wait_block, impl])


def run_checked(st: SyncState, nphase: int, mode: int, **kw) -> dict:
    """One launch with the counters reset first; returns the error / stale-read flags."""
    st.reset()
    launch_chain(st, nphase, mode, **kw)
    torch.cuda.synchronize()
    return {"err": int(st.err.item()), "bad": int(st.bad.item())}


def time_chain(st: SyncState, nphase: int, mode: int, reps: int = 25, **kw) -> float:
    """Median wall time (us) of one launch, measured with CUDA events around the kernel."""
    ts = []
    for _ in range(reps + 3):
        st.reset()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        launch_chain(st, nphase, mode, **kw)
        b.record()
        torch.cuda.synchronize()
        assert st.err.item() == 0 and st.bad.item() == 0, "sync benchmark flagged an error"
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts[3:])


def time_coop(grid: int, threads: int, nphase: int, work: int = 0, reps: int = 25) -> float:
    mod = module()
    f = mod.func("sync_chain_coop")
    vals = torch.zeros(nphase * grid * threads, dtype=torch.float32, device="cuda")
    bad = torch.zeros(1, dtype=torch.int32, device="cuda")
    ts = []
    for _ in range(reps + 3):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        cuda_rt.launch(f, grid, threads, [vals, bad, nphase, work], cooperative=True)
        b.record()
        torch.cuda.synchronize()
        assert bad.item() == 0
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts[3:])


def time_graph_nodes(grid: int, threads: int, nphase: int, work: int = 0, reps: int = 25) -> float:
    """N dependent tiny kernels in a CUDA graph (today's style); one replay timed with events."""
    mod = module()
    f = mod.func("tiny_phase")
    vals = torch.zeros(nphase * grid * threads, dtype=torch.float32, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for p in range(nphase):
            cuda_rt.launch(f, grid, threads, [vals, p, work])
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for p in range(nphase):
            cuda_rt.launch(f, grid, threads, [vals, p, work])
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps + 3):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts[3:])


def slope_us_per_phase(ns: list[int], ts: list[float]) -> tuple[float, float]:
    """Least squares t = a + b * n; returns (b, a)."""
    n = len(ns)
    mx, my = sum(ns) / n, sum(ts) / n
    b = sum((x - mx) * (y - my) for x, y in zip(ns, ts)) / sum((x - mx) ** 2 for x in ns)
    return b, my - b * mx
