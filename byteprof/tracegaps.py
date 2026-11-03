"""Split GPU idle time in a torch.profiler chrome trace into launch gaps and sync bubbles (pure).

A gap is the idle time between consecutive GPU kernels (kernels are merged first, so overlapping
kernels leave no gap). A gap is a *sync bubble* if a CPU-side synchronizing call overlaps it:
cudaStreamSynchronize, cudaDeviceSynchronize, cudaEventSynchronize, or a cudaMemcpy* call. A
cudaMemcpy* is counted whether or not it is device-to-host (the chrome trace carries no reliable
direction), and async copies that do not block the host are counted too, so the sync-bubble share
is an upper bound when the loop issues many async copies. Every other gap is a launch gap.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

GPU_KERNEL_CATS = {"kernel"}
SYNC_NAME = re.compile(r"^(cudaStreamSynchronize|cudaDeviceSynchronize|cudaEventSynchronize|cudaMemcpy)")
CPU_CATS = {"cuda_runtime", "cuda_driver"}


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def analyze_gaps(events: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """events: chrome-trace events (`ph == "X"` with ts/dur in microseconds). None if no GPU kernels."""
    kernels: list[tuple[float, float]] = []
    syncs: list[tuple[float, float]] = []
    for e in events:
        if e.get("ph") != "X" or "ts" not in e or "dur" not in e:
            continue
        span = (float(e["ts"]), float(e["ts"]) + float(e["dur"]))
        if e.get("cat") in GPU_KERNEL_CATS:
            kernels.append(span)
        elif e.get("cat") in CPU_CATS and SYNC_NAME.match(str(e.get("name", ""))):
            syncs.append(span)
    if not kernels:
        return None
    kernels.sort()
    syncs.sort()

    gaps: list[tuple[float, float]] = []
    cur_end = kernels[0][1]
    for s, e in kernels[1:]:
        if s > cur_end:
            gaps.append((cur_end, s))
        cur_end = max(cur_end, e)
    window = cur_end - kernels[0][0]

    bubble, launch = [], []
    j = 0
    for g0, g1 in gaps:
        while j < len(syncs) and syncs[j][1] <= g0:  # sync ended before this gap started
            j += 1
        # syncs are sorted by start; a later one may still start inside the gap
        k = j
        hit = False
        while k < len(syncs) and syncs[k][0] < g1:
            if syncs[k][1] > g0:
                hit = True
                break
            k += 1
        (bubble if hit else launch).append(g1 - g0)

    lengths = sorted(g1 - g0 for g0, g1 in gaps)
    total_idle = sum(lengths)
    return {
        "window_us": window,
        "kernels": len(kernels),
        "gaps": len(gaps),
        "idle_frac": total_idle / window if window else 0.0,
        "sync_bubble_frac": sum(bubble) / window if window else 0.0,
        "launch_gap_frac": sum(launch) / window if window else 0.0,
        "sync_bubbles": len(bubble),
        "launch_gaps": len(launch),
        "gap_median_us": _percentile(lengths, 0.5),
        "gap_p90_us": _percentile(lengths, 0.9),
    }
