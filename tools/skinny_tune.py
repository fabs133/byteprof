"""Stage 1 tuning for TASK_skinny_kernels.md: best (BN, BK, num_warps, num_stages) of the skinny kernels per
(kind, N, K, BM bucket), CUDA-graph timed on a ring of distinct weights. Writes byteprof/optim/skinny_cfg.json.

Usage: python -X utf8 tools/skinny_tune.py --kinds int8 [--shapes 2048x2048,...] [--bms 16,32,64] [--quick]
Existing entries for other (kind, shape, BM) cells are kept, so the sweep can be split into short runs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch

import skinny_bench as B
from byteprof.optim import skinny as S
from byteprof.quant.triton_int8 import quantize_rowwise_int8

CFG_FILE = Path(os.environ["SKINNY_CFG_OUT"]) if os.environ.get("SKINNY_CFG_OUT") else ROOT / "byteprof/optim/skinny_cfg.json"
BN_GRID = (8, 16, 32, 64)
BK_GRID = {"bf16": (64, 128, 256), "int8": (128, 256, 512), "int4": (128,)}  # int4: BN is the format tile
WARPS = (2, 4, 8)
STAGES = (2, 3, 4)
SPLITS = (2, 3, 4, 6, 8)  # tried on the best split-1 configuration only
REP_M = {16: 8, 32: 17, 64: 33}  # tuning row count per BM bucket


def candidates(kind: str, quick: bool):
    for bn, bk, nw, st, sp in itertools.product(BN_GRID if kind != "int4" else (32,), BK_GRID[kind], WARPS, STAGES, (1,)):
        if bn * bk > 16384 * (2 if nw == 8 else 1) // (1 if kind != "int4" else 2):
            continue
        if quick and (nw == 2 or st == 2):
            continue
        yield (bn, bk, nw, st, sp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="bf16,int8,int4")
    ap.add_argument("--shapes", default=",".join(f"{n}x{k}" for n, k in B.SHAPES))
    ap.add_argument("--bms", default="16,32,64")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    table = json.loads(CFG_FILE.read_text(encoding="utf-8")) if CFG_FILE.exists() else {}
    torch.manual_seed(0)
    for kind in a.kinds.split(","):
        for sh in a.shapes.split(","):
            n, k = (int(v) for v in sh.split("x"))
            ring = max(4, B.RING_BYTES // (n * k * 2))
            ws = [(torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(ring)]
            if kind == "bf16":
                packs = [(w, None, None) for w in ws]
            elif kind == "int8":
                packs = [(*quantize_rowwise_int8(w), None) for w in ws]
            else:
                packs = [S.quantize_int4(w, int(os.environ.get("SKINNY_ROWS", 32))) for w in ws]
            for bm in (int(v) for v in a.bms.split(",")):
                m = REP_M[bm]
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                best = None
                t0 = time.perf_counter()
                for cfg in candidates(kind, a.quick):
                    try:
                        def run(cfg=cfg):
                            for p in packs:
                                S._launch(kind, x, p[0], p[1], p[2], out, cfg)
                        t = min(B.time_graph(run, replays=20, trials=7)) / ring * 1000
                    except Exception as exc:  # e.g. too much shared memory for this tile
                        continue
                    if best is None or t < best[0]:
                        best = (t, cfg)
                base = best
                for sp in SPLITS:
                    cfg = (*base[1][:4], sp)
                    try:
                        def run(cfg=cfg):
                            for p in packs:
                                S._launch(kind, x, p[0], p[1], p[2], out, cfg)
                        t = min(B.time_graph(run, replays=20, trials=7)) / ring * 1000
                    except Exception:
                        continue
                    if t < best[0] * 0.97:  # a split must win by >3% (noise)
                        best = (t, cfg)
                table[f"{kind}|{n}|{k}|{bm}"] = list(best[1])
                print(f"{kind:5s} {n}x{k} BM={bm} (M={m}): best {best[0]:.1f} us cfg={best[1]} ({time.perf_counter() - t0:.0f}s)", flush=True)
                CFG_FILE.write_text(json.dumps(table, indent=1), encoding="utf-8")
            del ws, packs
            torch.cuda.empty_cache()
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
