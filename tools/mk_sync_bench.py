"""Stage 1 driver: us per synchronized phase by variant and N (TASK_megakernel_theory.md).

  python -X utf8 tools/mk_sync_bench.py [--out results/megakernel/sync.json] [--threads 256]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from byteprof.bench import cuda_rt, sync_bench as sb  # noqa: E402

NS = [10, 20, 40, 80]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/megakernel/sync.json")
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    grid = sb.sm_count()
    assert grid <= sb.max_coresident(a.threads)
    coop = bool(cuda_rt.device_attr(cuda_rt.ATTR_COOP))
    st = sb.SyncState(max(NS), grid, a.threads)
    variants: dict[str, dict] = {
        "none (work only)": dict(mode=3),
        "barrier (one counter), +__threadfence": dict(mode=0, impl=0),
        "barrier (one counter)": dict(mode=0, impl=1),
        "barrier, relaxed poll + fence": dict(mode=0, impl=3),
        "barrier, sync only (no read-back)": dict(mode=0, impl=5),
        "dataflow 1 producer": dict(mode=1, nprod=1, impl=1),
        "dataflow 2 producers": dict(mode=1, nprod=2, impl=1),
        "dataflow 4 producers": dict(mode=1, nprod=4, impl=1),
        "dataflow all blocks (flags)": dict(mode=1, nprod=min(grid, a.threads), impl=1),
        "dataflow 1 producer, sync only": dict(mode=1, nprod=1, impl=5),
        "dataflow 2 producers, sync only": dict(mode=1, nprod=2, impl=5),
        "dataflow 4 producers, sync only": dict(mode=1, nprod=4, impl=5),
    }
    res: dict = {"gpu": torch.cuda.get_device_name(0), "grid": grid, "threads": a.threads, "cooperative_launch": coop,
                 "cap_ns": sb.CAP_NS, "rows": {}}
    series: dict[str, dict[int, list[float]]] = {k: {n: [] for n in NS} for k in [*variants, "grid.sync()", "graph of tiny kernels"]}
    for rnd in range(a.rounds):
        for n in NS:
            for name, kw in variants.items():
                series[name][n].append(sb.time_chain(st, n, **kw))
            if coop:
                series["grid.sync()"][n].append(sb.time_coop(grid, a.threads, n))
            series["graph of tiny kernels"][n].append(sb.time_graph_nodes(grid, a.threads, n))
        print("round", rnd, "done", flush=True)
    for name, per_n in series.items():
        if not any(per_n[n] for n in NS):
            continue
        med = {n: sorted(v)[len(v) // 2] for n, v in per_n.items()}
        b, a0 = sb.slope_us_per_phase(NS, [med[n] for n in NS])
        res["rows"][name] = {"total_us": med, "us_per_phase_slope": b, "fixed_us": a0,
                             "us_per_phase_at_n": {n: med[n] / n for n in NS}, "all": per_n}
        print(f"{name:32s} " + " ".join(f"N={n}:{med[n]:7.1f}us" for n in NS) + f"  slope {b:6.3f} us/phase  fixed {a0:6.1f} us", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
