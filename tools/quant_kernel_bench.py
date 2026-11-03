"""Stage 1 of TASK_local_quant.md: do the weight-only kernels read fewer bytes at batch 1?

For every linear shape of a Bolmo local layer, builds a ring of RING distinct bf16 nn.Linear
modules (so the weights are not L2-resident across replays), quantizes copies with torchao,
captures one CUDA graph that applies all ring members to a single activation row, and reports
time per call and achieved GB/s (stored weight bytes / time).

Run: python -X utf8 tools/quant_kernel_bench.py [--compile] [--out results/quant/stage1.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

RING = 8
REPLAYS = 60
# (name, out, in, count per local layer)
SHAPES = [("q/k", 1024, 2048, 2), ("v/ogate/out_proj/proj", 2048, 2048, 3), ("mlp gate/up", 2816, 2048, 2),
          ("mlp down", 2048, 2816, 1)]


def make_ring(out_f: int, in_f: int, scheme: str, compiled: bool):
    from torchao.quantization import int4_weight_only, int8_weight_only, quantize_

    mods = []
    for _ in range(RING):
        lin = nn.Linear(in_f, out_f, bias=False, dtype=torch.bfloat16, device="cuda")
        nn.init.normal_(lin.weight, std=0.02)
        if scheme == "int8t":  # fused Triton int8 GEMV (byteprof/quant/triton_int8.py)
            from byteprof.quant.triton_int8 import gemv_int8, quantize_rowwise_int8

            q, s = quantize_rowwise_int8(lin.weight.data)

            class _Q(nn.Module):
                def forward(self, x, q=q, s=s):
                    return gemv_int8(x, q, s)

            mods.append(_Q())
            continue
        if scheme == "int8":
            quantize_(lin, int8_weight_only())
        elif scheme == "int4":
            quantize_(lin, int4_weight_only(group_size=128))
        mods.append(lin)
    return mods


def stored_bytes(lin: nn.Module, scheme: str, out_f: int, in_f: int) -> float:
    if scheme == "bf16":
        return out_f * in_f * 2
    if scheme in ("int8", "int8t"):
        return out_f * in_f + out_f * 2
    return out_f * in_f * 0.5 + out_f * (in_f // 128) * 4


def bench(scheme: str, out_f: int, in_f: int, compiled: bool) -> dict:
    mods = make_ring(out_f, in_f, scheme, compiled)
    x = torch.randn(1, in_f, dtype=torch.bfloat16, device="cuda")
    ref = mods[0]
    fns = [torch.compile(m, dynamic=False) for m in mods] if compiled else mods

    def run():
        y = x
        acc = None
        for m in fns:
            y = m(x)
            acc = y if acc is None else acc + y  # keep every result live, tiny extra op
        return acc

    with torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = run()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = run()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(5):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(REPLAYS):
                g.replay()
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b) / REPLAYS / RING * 1000)  # us per call
    us = sorted(times)[len(times) // 2]
    sb = stored_bytes(ref, scheme, out_f, in_f)
    return {"us_per_call": us, "min_us": min(times), "max_us": max(times), "stored_bytes": sb,
            "gb_per_s": sb / (us * 1e-6) / 1e9, "check": float(out.float().abs().mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", action="store_true", help="torch.compile each quantized module")
    ap.add_argument("--out", default="results/quant/stage1.json")
    ap.add_argument("--schemes", default="bf16,int8,int8t,int4")
    args = ap.parse_args()
    import torchao

    res: dict = {"torch": torch.__version__, "torchao": torchao.__version__, "gpu": torch.cuda.get_device_name(0),
                 "compile": args.compile, "ring": RING, "rows": {}}
    for scheme in args.schemes.split(","):
        for name, o, i, n in SHAPES:
            key = f"{scheme}|{name}"
            try:
                t0 = time.perf_counter()
                res["rows"][key] = {"out": o, "in": i, "count": n, **bench(scheme, o, i, args.compile and scheme != "bf16")}
                r = res["rows"][key]
                print(f"{scheme:5s} {name:22s} {o}x{i}: {r['us_per_call']:7.1f} us  {r['gb_per_s']:6.1f} GB/s "
                      f"(min {r['min_us']:.1f} max {r['max_us']:.1f}; {time.perf_counter()-t0:.0f}s)", flush=True)
            except Exception as exc:  # a failed scheme is a result
                res["rows"][key] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
                print(f"{scheme:5s} {name:22s} FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
