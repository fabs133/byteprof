"""Stage 2 driver: step time and achieved GB/s of the local-stack chain, per-phase graph vs persistent kernel.

  python -X utf8 tools/mk_stream_bench.py --q bf16 [--rounds 5] [--out results/megakernel/stream-bf16.json]
Variants (a: one kernel per phase in a CUDA graph; b: persistent, waits for the dependency before loading;
c: persistent, stages the next phase's first rows in shared memory and preloads the mLSTM state before the barrier).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from byteprof.bench import stream_bench as sbn  # noqa: E402

STATE_BYTES = 5 * 2 * 16 * 64 * 128 * 4   # c read + write per step (the other state is tiny)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="bf16", choices=["bf16", "int8", "int4"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.out or f"results/megakernel/stream-{a.q}.json")
    bw = sbn.copy_bandwidth()
    print("bandwidth", bw, flush=True)
    st = sbn.Stack(a.q)
    grid = torch.cuda.get_device_properties(0).multi_processor_count
    variants: dict[str, tuple[callable, int]] = {}
    for g in (38, 76, 114, 152, 228):
        variants[f"a: per-phase graph, {g} blocks"] = (lambda g=g: st.per_phase(g), 0)
    for name, sw, pre in (("b: persistent, no prefetch", 80 * 1024, 0),
                          ("c: persistent, prefetch 80 KiB (end of phase)", 80 * 1024, 1),
                          ("c: persistent, prefetch 32 KiB (end of phase)", 32 * 1024, 1),
                          ("c2: persistent, double-buffered prefetch 80 KiB", 80 * 1024, 2),
                          ("c0: persistent, state preload only", 0, 1)):
        variants[name] = (lambda pre=pre: st.mega(pre), sw)
    graphs = {}
    for name, (fn, sw) in variants.items():
        st.sw_bytes = sw
        st.reset_state()
        graphs[name] = st.graph_of(fn)
        assert int(st.ctl[2].item()) == 0, name
    series = {k: [] for k in graphs}
    for r in range(a.rounds):
        for name, g in graphs.items():
            st.sw_bytes = variants[name][1]
            series[name].append(sbn.time_graph(g, reps=20, trials=5))
        print("round", r, flush=True)
    assert int(st.ctl[2].item()) == 0, "error flag set during timing"
    res = {"q": a.q, "weight_bytes": st.weight_bytes, "state_bytes": STATE_BYTES, "bandwidth": bw, "variants": {}}
    for name, v in series.items():
        t = statistics.median(v)
        res["variants"][name] = {"us_median": t, "us_all": v, "weight_GBs": st.weight_bytes / t / 1e3,
                                 "frac_copy": st.weight_bytes / t / 1e3 / bw["copy_rw_GBs"],
                                 "frac_read": st.weight_bytes / t / 1e3 / bw["read_only_GBs"],
                                 "all_traffic_GBs": (st.weight_bytes + STATE_BYTES) / t / 1e3}
        print(f"{name:40s} {t:8.1f} us  weights {st.weight_bytes / t / 1e3:6.1f} GB/s  "
              f"{100 * res['variants'][name]['frac_copy']:5.1f}% of copy  {100 * res['variants'][name]['frac_read']:5.1f}% of read", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
