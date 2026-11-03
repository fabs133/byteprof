"""Stage 0 of TASK_megakernel_theory.md: census of a graphed local round (A_K + C_K) from tools/nsys_rounds.py traces.

For each traced block (K, reps back-to-back rounds): kernels per round, duration distribution, kernel time by class,
first-to-last span per round, and the split of the excess over the weight-read floor into
  inside kernels  = kernel sum - floor        (short-kernel ramp/tail, low occupancy, non-matmul kernels)
  between kernels = span - kernel sum         (gaps between kernel nodes and between graph launches)
Usage: python -I tools/mk_census.py results/megakernel/nsys/bf16sk.sqlite --weight-bytes 360.35e6 --bw 411 \
         --out results/megakernel/census-bf16sk.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from nsys_rounds_analyze import classify  # noqa: E402


MATMUL = ("_skinny_kernel", "tinygemm", "gemm", "Kernel2", "gemv", "splitK", "xmma", "cutlass")


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def census(sqlite_path: str, weight_bytes: float, bw_gbs: float, k_nonvtx: int | None = None, warm: int = 300,
           reps_nonvtx: int = 30) -> dict:
    c = sqlite3.connect(sqlite_path)
    sid = dict(c.execute("select id, value from StringIds"))
    has_nvtx = c.execute("select count(*) from sqlite_master where name = 'NVTX_EVENTS'").fetchone()[0]
    rng = c.execute("select start, end, text from NVTX_EVENTS where text like 'block:%' order by start").fetchall() if has_nvtx else []
    kern = c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    floor_us = weight_bytes / (bw_gbs * 1e9) * 1e6
    res = {"weight_bytes": weight_bytes, "bw_GBs": bw_gbs, "floor_us": floor_us, "blocks": {}}
    if not rng:   # trace without NVTX: the capture holds `warm` + `reps_nonvtx` identical rounds of one K
        assert k_nonvtx is not None and len(kern) % (warm + reps_nonvtx) == 0, (len(kern), warm, reps_nonvtx)
        n_ = len(kern) // (warm + reps_nonvtx)
        tail = kern[-reps_nonvtx * n_:]
        rng = [(tail[0][0], tail[-1][1], f"block:{k_nonvtx}:{reps_nonvtx}")]
    for s, e, text in rng:
        _, k_, reps_ = text.split(":")
        k, reps = int(k_), int(reps_)
        ks = [(a, b, sid.get(n, str(n))) for a, b, n in kern if s <= a <= e]
        assert len(ks) % reps == 0
        n = len(ks) // reps
        # steady state: drop the first round (clock ramp), average the others
        rounds = [ks[i * n:(i + 1) * n] for i in range(reps)]
        durs = [[(b - a) / 1000 for a, b, _ in r] for r in rounds]
        spans = [(r[-1][1] - r[0][0]) / 1000 for r in rounds]
        # span of one round measured start-to-start of consecutive rounds (includes the gap to the next round)
        period = (rounds[-1][0][0] - rounds[1][0][0]) / 1000 / (reps - 2)
        first = durs[1]
        cls: dict[str, list[float]] = {}
        for (a, b, nm), d in zip(rounds[1], first):
            cls.setdefault(classify(nm), []).append(d)
        ksum = sum(sum(d) for d in durs[1:]) / (reps - 1)
        span = sum(spans[1:]) / (reps - 1)
        names: dict[str, list[float]] = {}
        for (a, b, nm), d in zip(rounds[1], first):
            names.setdefault(nm.split("<")[0][:60], []).append(d)
        mm = [d for (a, b, nm), d in zip(rounds[1], first) if any(t in nm for t in MATMUL)]
        top = sorted(((sum(v), len(v), nm) for nm, v in names.items()), reverse=True)[:8]
        res["blocks"][k] = {
            "reps": reps, "kernels": n, "kernel_sum_us": ksum, "span_us": span, "period_us": period,
            "inside_excess_us": ksum - floor_us, "between_us": period - ksum,
            "excess_us": period - floor_us,
            "share_above_floor": (period - floor_us) / period,
            "share_of_excess_inside": (ksum - floor_us) / (period - floor_us),
            "matmul_kernels": len(mm), "matmul_us": sum(mm), "other_kernels": n - len(mm), "other_us": sum(first) - sum(mm),
            "dur_p50": pct(first, 0.5), "dur_p90": pct(first, 0.9), "dur_max": max(first),
            "n_lt5": sum(1 for d in first if d < 5), "n_lt10": sum(1 for d in first if d < 10),
            "us_lt5": sum(d for d in first if d < 5), "us_lt10": sum(d for d in first if d < 10),
            "classes": {k_: {"count": len(v), "us": sum(v)} for k_, v in cls.items()},
            "top_kernels": [{"name": nm, "count": cnt, "us": us} for us, cnt, nm in top],
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--weight-bytes", type=float, required=True)
    ap.add_argument("--bw", type=float, default=411.0)
    ap.add_argument("--out")
    ap.add_argument("--k", type=int, help="trace without NVTX ranges: the K of the single traced block")
    a = ap.parse_args()
    r = census(a.sqlite, a.weight_bytes, a.bw, k_nonvtx=a.k)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"floor {r['floor_us']:.0f} us ({a.weight_bytes / 1e6:.1f} MB at {a.bw} GB/s)")
    for k, b in r["blocks"].items():
        print(f"K={k}: {b['kernels']} kernels, kernel sum {b['kernel_sum_us']:.0f} us, period {b['period_us']:.0f} us, "
              f"excess {b['excess_us']:.0f} us = {100 * b['share_above_floor']:.0f}% of the round; inside {b['inside_excess_us']:.0f} "
              f"({100 * b['share_of_excess_inside']:.0f}%), between {b['between_us']:.0f}; <10us: {b['n_lt10']} kernels {b['us_lt10']:.0f} us")
        print(f"   matmul kernels {b['matmul_kernels']}: {b['matmul_us']:.0f} us (floor {r['floor_us']:.0f}); other {b['other_kernels']}: {b['other_us']:.0f} us")
        print("   classes:", {k_: (v['count'], round(v['us'])) for k_, v in b["classes"].items()})


if __name__ == "__main__":
    main()
