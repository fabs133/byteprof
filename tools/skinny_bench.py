"""Stages 0 and 1 of TASK_skinny_kernels.md: weight-only matmul kernels at M = 1..33 inside CUDA graphs.

For every distinct linear shape (out, in) of Bolmo and OLMo 2 and every M, a ring of RING distinct weight sets
(so the weights are not L2-resident across replays) is applied to one activation block in a CUDA graph; the
time per call is the median over trials. Backends:

  bf16_cublas   F.linear on bf16 weights                                   (stage 0)
  int8_cur      today's Int8Linear: Triton GEMV at M = 1, dequantize + F.linear for M > 1 (dequantize timed too)
  int4_tiny     torchao int4_weight_only(group 128), tinygemm
  bf16_skinny / int8_skinny / int4_skinny   byteprof.optim.skinny          (stage 1)

Also the device copy bandwidth (large device-to-device copy) as the bandwidth reference.
Usage: python -X utf8 tools/skinny_bench.py --stage 0|1 [--out results/skinny/stageN.json] [--ms 1,2,...] [--shapes ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

MS = (1, 2, 3, 4, 5, 8, 9, 16, 17, 33)
# (out, in): global layers (and OLMo 2's blocks, same shapes) and local layers
SHAPES = [(2048, 2048), (8192, 2048), (2048, 8192), (1024, 2048), (2816, 2048), (2048, 2816)]
RING_BYTES = 96 * 2**20  # bf16 weight bytes in the ring (L2 is 3 MB)


def stored_bytes(kind: str, n: int, k: int) -> float:
    if kind.startswith("bf16"):
        return n * k * 2
    if kind.startswith("int8"):
        return n * k + n * 2
    return n * k * 0.5 + n * (k // 128) * 4


def time_graph(run, replays: int = 30, trials: int = 5) -> list[float]:
    """Capture run() in a CUDA graph; return ms per replay for each trial."""
    with torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        out = []
        for _ in range(trials):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(replays):
                g.replay()
            b.record()
            torch.cuda.synchronize()
            out.append(a.elapsed_time(b) / replays)
        del g
        return out


def copy_bandwidth() -> float:
    """GB/s of a large device-to-device copy (read + write counted, as nvidia's bandwidthTest does)."""
    n = 256 * 2**20
    a = torch.empty(n, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    t = statistics.median(time_graph(lambda: b.copy_(a), replays=10, trials=7))
    del a, b
    return 2 * n / (t * 1e-3) / 1e9


def make_members(backend: str, n: int, k: int, ring: int):
    """List of callables f(x) -> y, one per distinct weight set, plus a reference function for checking."""
    from byteprof.optim import skinny as S
    from byteprof.quant.triton_int8 import gemv_int8, quantize_rowwise_int8

    members = []
    for _ in range(ring):
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        if backend == "bf16_cublas":
            members.append(lambda x, w=w: F.linear(x, w))
        elif backend == "bf16_skinny":
            members.append(lambda x, w=w: S.skinny_mm(x, w))
        elif backend in ("int8_cur", "int8_skinny"):
            q, sc = quantize_rowwise_int8(w)
            if backend == "int8_skinny":
                members.append(lambda x, q=q, sc=sc: S.skinny_mm(x, q, sc))
            else:
                def cur(x, q=q, sc=sc):
                    if x.shape[0] == 1:
                        return gemv_int8(x, q, sc)
                    return F.linear(x, q.to(x.dtype) * sc.to(x.dtype)[:, None])  # Int8Linear's M > 1 path
                members.append(cur)
        elif backend == "int4_tiny":
            import torch.nn as nn
            from torchao.quantization import int4_weight_only, quantize_

            lin = nn.Linear(k, n, bias=False, dtype=torch.bfloat16, device="cuda")
            lin.weight.data.copy_(w)
            quantize_(lin, int4_weight_only(group_size=128))
            members.append(lin)
        elif backend == "int4_skinny":
            p, sc, z = S.quantize_int4(w)
            members.append(lambda x, p=p, sc=sc, z=z: S.skinny_mm(x, p, sc, z))
        else:
            raise ValueError(backend)
        del w
    return members


def bench_cell(backend: str, n: int, k: int, ms: tuple[int, ...], copy_gbs: float) -> dict:
    ring = max(4, RING_BYTES // (n * k * 2))
    members = make_members(backend, n, k, ring)
    kind = backend.split("_")[0]
    sb = stored_bytes(kind, n, k)
    res = {}
    for m in ms:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        trials = time_graph(lambda: [f(x) for f in members])
        us = [t / ring * 1000 for t in trials]
        med = statistics.median(us)
        res[str(m)] = {"us": med, "min_us": min(us), "max_us": max(us), "gbs": sb / (med * 1e-6) / 1e9,
                       "floor_us": sb / (copy_gbs * 1e9) * 1e6}
    del members
    torch.cuda.empty_cache()
    return {"n": n, "k": k, "stored_bytes": sb, "ring": ring, "m": res}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--backends", default=None)
    a = ap.parse_args()
    backends = (a.backends or ("bf16_cublas,int8_cur,int4_tiny" if a.stage == 0 else "bf16_skinny,int8_skinny,int4_skinny")).split(",")
    ms = tuple(int(x) for x in a.ms.split(","))
    out = Path(a.out or ROOT / f"results/skinny/stage{a.stage}.json")
    import triton

    res: dict = {"torch": torch.__version__, "triton": triton.__version__, "gpu": torch.cuda.get_device_name(0),
                 "ring_bytes": RING_BYTES, "ms": ms, "cells": {}}
    torch.manual_seed(0)
    res["copy_gbs"] = copy_bandwidth()
    print(f"device copy bandwidth {res['copy_gbs']:.1f} GB/s (read+write)", flush=True)
    for be in backends:
        for n, k in SHAPES:
            t0 = time.perf_counter()
            try:
                res["cells"][f"{be}|{n}x{k}"] = bench_cell(be, n, k, ms, res["copy_gbs"])
                c = res["cells"][f"{be}|{n}x{k}"]["m"]
                print(f"{be:12s} {n}x{k}: " + "  ".join(f"M{m}={c[str(m)]['us']:.1f}" for m in ms)
                      + f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
            except Exception as exc:  # a failed backend is a result
                res["cells"][f"{be}|{n}x{k}"] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
                print(f"{be} {n}x{k} FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
