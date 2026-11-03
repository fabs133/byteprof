"""Stage 3 of TASK_megakernel_theory.md: predicted local-megakernel step times, predicted ms/byte of the existing cells,
the fair floor ratio (H64) and the gate.

Inputs (all measured earlier in this task or in the earlier tasks, see FINDINGS.md):
  results/megakernel/{sync,asym,tsprof-bf16}.json, stream-{bf16,int8,int4}.json   (Stages 1-2)
  results/megakernel/census-{bf16sk,int4}.json                                      (Stage 0)
  results/skinny/bench/g8/bench.json, results/parallel/bench/int4-fused-L0/bench.json  (today's local rounds A_K + C_K)
  results/skinny/speed_report.json                                                  (measured ms/byte, rounds/byte)
  results/p4/tallies_{bf16,int4}.json                                               (share of K = 0 rounds)
Output: results/megakernel/costmodel.{json,md}.   python -X utf8 tools/mk_costmodel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results/megakernel"
BW_COPY = 411.0   # GB/s, the task's reference (copy, read + write counted)

N_PHASE = 28            # phases of the modelled local round
EXTRA_PHASES = 3        # embedding gather + add, argmax / token write, boundary bookkeeping: not in the benchmark
PARAMS = dict(local=180.17e6, bolmo_global=1.0737e9, olmo_blocks=1.0737e9, olmo_head=0.2097e9)
BYTES_PER_PARAM = {"bf16": 2.0, "int8": 1.0, "int4": 0.53125}   # int4: 4 bit + fp16 scale + fp16 zero per 128
STATE_BYTES_PER_ROUND = 5 * 2 * 16 * 64 * 128 * 4 + 5 * 2 * 16 * 64 * 4   # c and n, read + write


def load(p: str):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def best_persistent(stream: dict) -> tuple[str, float]:
    items = [(k, v["us_median"]) for k, v in stream["variants"].items() if k.startswith(("b:", "c"))]
    return min(items, key=lambda kv: kv[1])


def step_models() -> dict:
    sync = load("results/megakernel/sync.json")["rows"]
    sync_only = sync["barrier, sync only (no read-back)"]["us_per_phase_slope"]
    sync_full = sync["barrier (one counter)"]["us_per_phase_slope"]
    asym = load("results/megakernel/asym.json")
    ts = load("results/megakernel/tsprof-bf16.json")
    out: dict = {"sync_us_per_phase": sync_only, "sync_with_readback_us": sync_full, "asym_GBs": asym["fit_GBs"],
                 "fixed_per_phase_us": asym["fit_fixed_us"], "mlstm_phase_us": ts["mlstm_phase_us"],
                 "extra_us": EXTRA_PHASES * (ts["mlstm_phase_us"] * 0.5 + sync_only)}
    fixed_bf16 = None
    for q in ("bf16", "int8", "int4"):
        st = load(f"results/megakernel/stream-{q}.json")
        name, t = best_persistent(st)
        a_name = min(((k, v["us_median"]) for k, v in st["variants"].items() if k.startswith("a:")), key=lambda kv: kv[1])
        wb = st["weight_bytes"]
        stream_us = wb / (asym["fit_GBs"] * 1e3)
        n_mlstm = 5
        ideal = stream_us + (N_PHASE - 1) * sync_only + n_mlstm * ts["mlstm_phase_us"]
        out[q] = {"weight_bytes": wb, "persistent_best": name, "measured_persistent_us": t, "measured_graph_us": a_name[1],
                  "graph_best": a_name[0], "stream_floor_us": stream_us, "ideal_overlap_us": ideal,
                  "measured_GBs": wb / t / 1e3, "frac_of_copy": wb / t / 1e3 / BW_COPY}
        if q == "bf16":
            fixed_bf16 = t - stream_us
            out["fixed_us_bf16"] = fixed_bf16
    # extrapolation by bytes for the instruction-bound int8 / int4 kernels (bandwidth-bound kernel assumed)
    for q in ("int8", "int4"):
        out[q]["extrapolated_us"] = out[q]["stream_floor_us"] + fixed_bf16
    return out


def todays_rounds() -> dict:
    g8 = load("results/skinny/bench/g8/bench.json")["rounds"]
    i4 = load("results/parallel/bench/int4-fused-L0/bench.json")["rounds"]
    bf = load("results/parallel/bench/bf16-fused-L0/bench.json")["rounds"]
    return {"g8": {k: v["ms"] for k, v in g8.items()}, "int4": {k: v["ms"] for k, v in i4.items()},
            "bf16": {k: v["ms"] for k, v in bf.items()}}


def k0_share(prec: str) -> float:
    t = load(f"results/p4/tallies_{prec}.json")["cells"]
    key = "default|P4|16|4" if prec == "int4" else "default|P4|32|6"
    c = t[key]["by_kp"]
    tot = sum(c.values())
    return sum(n for k, n in c.items() if k.split(",")[0] == "0") / tot


def predictions(sm: dict) -> dict:
    cells = load("results/skinny/speed_report.json")["cells"]
    pair0 = todays_rounds()
    res: dict = {}
    settings = [
        # name, speed cells (minimal, P4), today's K=0 pair, megakernel precision for the local stack
        ("near-lossless (global int8, local bf16 skinny)", "B1|bolmo-g8-none", "B1|bolmo-g8-p4", pair0["g8"]["0"], "bf16", "bf16"),
        ("all-int4 (tinygemm)", "B3|int4-none", "B3|int4-p4", pair0["int4"]["0"], "int4", "int4"),
    ]
    for name, cmin, cp4, pair_k0, mprec, tprec in settings:
        f0 = k0_share("bf16" if mprec == "bf16" else "int4")
        r: dict = {"today_pair_k0_ms": pair_k0, "k0_share_of_rounds": f0}
        for label, t_us in (("measured kernel", sm[mprec]["measured_persistent_us"]),
                            ("ideal overlap", sm[mprec]["ideal_overlap_us"]),
                            ("extrapolated by bytes", sm[mprec].get("extrapolated_us", sm[mprec]["measured_persistent_us"]))):
            t_ms = (t_us + sm["extra_us"]) / 1000
            delta = pair_k0 - t_ms
            row = {"mega_k0_ms": t_ms, "delta_per_round_ms": delta}
            for loop, cell in (("minimal", cmin), ("P4", cp4)):
                ms, rpb = cells[cell]["mean"], cells[cell]["rounds_per_byte"] or 1.0
                central = ms - rpb * delta
                low = ms - rpb * (f0 * delta + (1 - f0) * 0.5 * delta)
                row[loop] = {"today_ms_per_byte": ms, "rounds_per_byte": rpb, "pred_ms_per_byte": central,
                             "gain_pct": 100 * (ms - central) / ms, "pred_low_ms_per_byte": low,
                             "gain_low_pct": 100 * (ms - low) / ms}
            r[label] = row
        res[name] = r
    return res


def floor_ratio() -> dict:
    """H64: both models at the weight-read floor, from real parameter counts and measured passes / rounds per byte."""
    cells = load("results/skinny/speed_report.json")["cells"]
    # measured loop statistics (FINDINGS.md): bolmo bytes per global pass, rounds per byte; olmo passes per byte
    stats = {
        "minimal": {"bolmo_global_per_byte": 1 / 3.38, "bolmo_rounds_per_byte": 1.0, "olmo_passes_per_byte": 0.275},
        "drafted bf16": {"bolmo_global_per_byte": 1 / 7.4, "bolmo_rounds_per_byte": cells["B1|bf16-p4"]["rounds_per_byte"],
                         "olmo_passes_per_byte": cells["B4|olmo-bf16-combined-k8"]["rounds_per_byte"]},
        "drafted int8": {"bolmo_global_per_byte": 1 / 6.6, "bolmo_rounds_per_byte": cells["B1|bolmo-g8-p4"]["rounds_per_byte"],
                         "olmo_passes_per_byte": cells["B4|olmo-int8sk-k8"]["rounds_per_byte"]},
    }

    def bolmo_gb(loop, pg, pl):
        s = stats[loop]
        return (s["bolmo_global_per_byte"] * PARAMS["bolmo_global"] * BYTES_PER_PARAM[pg]
                + s["bolmo_rounds_per_byte"] * (PARAMS["local"] * BYTES_PER_PARAM[pl] + STATE_BYTES_PER_ROUND)) / 1e9

    def olmo_gb(loop, p, head=None):
        s = stats[loop]
        return s["olmo_passes_per_byte"] * (PARAMS["olmo_blocks"] * BYTES_PER_PARAM[p]
                                            + PARAMS["olmo_head"] * BYTES_PER_PARAM[head or p]) / 1e9

    rows = []
    for loop in stats:
        for prec in ("bf16", "int8", "int4"):
            if loop == "drafted bf16" and prec != "bf16" or loop == "drafted int8" and prec != "int8":
                continue
            b_same, b_l4 = bolmo_gb(loop, prec, prec), bolmo_gb(loop, prec, "int4")
            o = olmo_gb(loop, prec)
            o_headbf = olmo_gb(loop, prec, "bf16")
            rows.append({"loop": loop, "precision": prec, "bolmo_GB_per_byte": b_same, "bolmo_local_int4_GB_per_byte": b_l4,
                         "olmo_GB_per_byte": o, "olmo_head_bf16_GB_per_byte": o_headbf,
                         "ratio_equal_precision": b_same / o, "ratio_local_int4": b_l4 / o,
                         "ratio_equal_precision_head_bf16": b_same / o_headbf, "ratio_local_int4_head_bf16": b_l4 / o_headbf,
                         "bolmo_floor_ms_per_byte": b_same / BW_COPY * 1e3, "olmo_floor_ms_per_byte": o / BW_COPY * 1e3})
    return {"stats": stats, "rows": rows}


def main() -> int:
    sm = step_models()
    pred = predictions(sm)
    fr = floor_ratio()
    out = {"step_models": sm, "predictions": pred, "floor_ratio": fr}
    # gate
    gate = {}
    for name, p in pred.items():
        best = p["measured kernel"]["P4"]
        gate[name] = {"pred_gain_pct_measured_kernel": best["gain_pct"], "pred_gain_pct_ideal": p["ideal overlap"]["P4"]["gain_pct"],
                      "pred_gain_pct_extrapolated": p["extrapolated by bytes"]["P4"]["gain_pct"]}
    out["gate"] = gate
    (R / "costmodel.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# Stage 3: megakernel cost model", "",
          "## Step models (local K = 0 round, microseconds)", "",
          "| precision | weight MB | today's graph of 28 phase kernels | persistent (best variant) | stream floor at asymptote | ideal overlap (floor + sync + mLSTM) | extrapolated by bytes |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for q in ("bf16", "int8", "int4"):
        m = sm[q]
        md.append(f"| {q} | {m['weight_bytes'] / 1e6:.1f} | {m['measured_graph_us']:.0f} | {m['measured_persistent_us']:.0f} | {m['stream_floor_us']:.0f} | "
                  f"{m['ideal_overlap_us']:.0f} | {m.get('extrapolated_us', float('nan')):.0f} |")
    md += ["", f"sync {sm['sync_us_per_phase']:.2f} us/phase (sync only), asymptote {sm['asym_GBs']:.0f} GB/s, fixed cost per phase kernel {sm['fixed_per_phase_us']:.1f} us, "
           f"mLSTM phase {sm['mlstm_phase_us']:.1f} us, unmodelled extras {sm['extra_us']:.0f} us.", ""]
    for name, p in pred.items():
        md += [f"## {name}", "", f"Today's measured K = 0 local round (A + C): {p['today_pair_k0_ms']:.3f} ms; K = 0 share of P4 rounds {100 * p['k0_share_of_rounds']:.1f}%.", "",
               "| megakernel step | step ms (+extras) | saving per round ms | minimal loop ms/byte (gain) | P4 ms/byte (gain) | P4 gain if the saving halves for K > 0 |", "|---|---:|---:|---:|---:|---:|"]
        for label in ("measured kernel", "ideal overlap", "extrapolated by bytes"):
            r = p[label]
            md.append(f"| {label} | {r['mega_k0_ms']:.3f} | {r['delta_per_round_ms']:+.3f} | {r['minimal']['pred_ms_per_byte']:.3f} ({r['minimal']['gain_pct']:+.1f}%) | "
                      f"{r['P4']['pred_ms_per_byte']:.3f} ({r['P4']['gain_pct']:+.1f}%) | {r['P4']['gain_low_pct']:+.1f}% |")
        md.append("")
    md += ["## Fair floor ratio (H64)", "", "Both models at the weight-read floor (bytes read per generated byte / 411 GB/s); OLMo 2 head at the same precision as its blocks unless stated.", "",
           "| loop | precision | Bolmo GB/byte | OLMo GB/byte | ratio, equal precision | ratio, Bolmo local int4 | ratio equal, OLMo head bf16 | ratio local int4, OLMo head bf16 | Bolmo floor ms/byte | OLMo floor ms/byte |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in fr["rows"]:
        md.append(f"| {r['loop']} | {r['precision']} | {r['bolmo_GB_per_byte']:.3f} | {r['olmo_GB_per_byte']:.3f} | {r['ratio_equal_precision']:.2f}x | {r['ratio_local_int4']:.2f}x | "
                  f"{r['ratio_equal_precision_head_bf16']:.2f}x | {r['ratio_local_int4_head_bf16']:.2f}x | {r['bolmo_floor_ms_per_byte']:.3f} | {r['olmo_floor_ms_per_byte']:.3f} |")
    (R / "costmodel.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
