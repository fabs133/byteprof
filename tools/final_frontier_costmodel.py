"""Stage 0 of TASK_final_frontier.md: predicted local round cost vs K and P4 ms/byte for configurations A and B.

A = local int8 + global int8 (skinny kernels), B = local int4 + global int8 (int4 via tinygemm for small M and the skinny
int4 kernel where it wins, dispatch by M, per shape). Reference = global int8 + bf16 local (skinny).

Same machinery as tools/skinny_costmodel.py: the measured pieces of the P4 cost model (verify pair per K, global pass per
P) with the linear-kernel part swapped using the Stage 0/1 kernel tables (results/skinny/stage{0,1}.json). Two acceptance
frames are used because the text of a quantized model drafts differently and the model cannot predict that: the bf16 P4
tallies (bf16 pieces as baseline) and the int4 P4 tallies (int4 pieces as baseline). Predictions are expressed as ratios to the
reference variant in the same frame and then applied to measured cells (reference g8 P4 1.588 ms/byte, measured g8l4 P4
1.430 ms/byte for the skinny-only int4 variant).

Usage: python tools/final_frontier_costmodel.py [--out results/final_frontier]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import aligned_costmodel as acm  # noqa: E402
import p4_costmodel as p4  # noqa: E402
import skinny_costmodel as sk  # noqa: E402

REF_MS = 1.588  # measured g8 P4, default prompts, 256 new bytes (FINDINGS.md, Skinny kernels)
G8L4_MS = 1.430  # measured g8l4 P4 (global int8 + local int4, skinny int4 everywhere)
OLMO_I8_MS = 0.994  # measured OLMo 2 int8 skinny combined K = 8 (same session as the g8 reference)
LABEL_TINY, LABEL_SKINNY = "int4", "int4s"


class DispatchTable(sk.Table):
    """Adds the label 'int4d': per shape and row count the faster of tinygemm and the skinny int4 kernel."""

    def us(self, label: str, n: int, k: int, m: int) -> float:
        if label == "int4d":
            return min(super().us(LABEL_TINY, n, k, m), super().us(LABEL_SKINNY, n, k, m))
        return super().us(label, n, k, m)


def crossover(t: DispatchTable) -> dict:
    out = {}
    for n, k, _ in sk.LOCAL_LAYER:
        out[f"{n}x{k}"] = [m for m in range(1, 34) if t.us(LABEL_SKINNY, n, k, m) < t.us(LABEL_TINY, n, k, m)][:1]
    return out


def variants() -> list[tuple[str, str, str, tuple[str, str], tuple[str, str]]]:
    """(key, name, frame, local (baseline, new), global (baseline, new))."""
    rows = []
    for frame, bl, bg in (("bf16", "bf16", "bf16"), ("int4", "int4", "int4")):
        for key, name, new_local in (
            ("ref", "reference: global int8 + local bf16 (skinny)", "bf16s"),
            ("A", "A: local int8 + global int8 (skinny)", "int8s"),
            ("B_skinny", "B': local int4 skinny + global int8 (= measured g8l4)", "int4s"),
            ("B_tiny", "B'': local int4 tinygemm + global int8", "int4"),
            ("B", "B: local int4 dispatch (tinygemm small M, skinny large M) + global int8", "int4d"),
        ):
            rows.append((key, name, frame, (bl, new_local), (bg, "int8s")))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/final_frontier")
    a = ap.parse_args()
    s0 = json.loads((ROOT / "results/skinny/stage0.json").read_text(encoding="utf-8"))
    s1 = json.loads((ROOT / "results/skinny/stage1.json").read_text(encoding="utf-8"))
    t = DispatchTable(s0, s1)
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"frames": {}, "crossover_first_m_where_skinny_wins": crossover(t)}
    ks = (0, 1, 2, 4, 6, 8, 12, 16, 24, 32)
    local_ms = {lab: {k: sk.local_pass_us(t, lab, k + 1) / 1000 for k in ks} for lab in ("bf16", "bf16s", "int8s", "int4", "int4s", "int4d")}
    glob_ms = {lab: {p: sk.global_pass_us(t, lab, p) / 1000 for p in (1, 2, 4, 8)} for lab in ("bf16s", "int8s")}
    res["local_linears_ms"], res["global_linears_ms"] = local_ms, glob_ms
    for frame in ("bf16", "int4"):
        ex = p4.load_extra(frame)
        fr: dict = {}
        for key, name, fm, loc, glo in variants():
            if fm != frame:
                continue
            v = sk.Variant(name, frame, loc, glo)
            I = sk.adjusted_inputs(frame, t, v)
            cell = sk.best_cell(frame, I, ex, acm.RETOK_US)
            fr[key] = {"name": name, "k_cross": cell["k_cross"], "k_within": cell["k_within"],
                       "ext_ms": cell["ext"]["ms_per_byte"], "default_ms": cell["default"]["ms_per_byte"],
                       "bytes_per_round": cell["default"]["bytes_per_round"], "bytes_per_global": cell["default"]["bytes_per_global"],
                       "rounds_per_byte": cell["default"]["rounds_per_byte"],
                       "pair_ms": {str(k): I.pair[k] for k in ks if k in I.pair}, "B_ms": I.B}
        res["frames"][frame] = fr
    # predictions applied to measured cells
    pred: dict = {}
    for key in ("A", "B", "B_tiny", "B_skinny"):
        per_frame = {}
        for frame in ("bf16", "int4"):
            fr = res["frames"][frame]
            per_frame[frame] = fr[key]["default_ms"] / fr["ref"]["default_ms"]
        mean_ratio = sum(per_frame.values()) / 2
        pred[key] = {"model_ratio_to_ref": per_frame, "ms_from_ref_anchor": {f: REF_MS * r for f, r in per_frame.items()},
                     "ms_mean": REF_MS * mean_ratio, "gain_vs_ref_pct": (1 - mean_ratio) * 100}
    # B anchored on the measured g8l4 (same text effect): g8l4 * model(B)/model(B')
    anchors = {}
    for frame in ("bf16", "int4"):
        fr = res["frames"][frame]
        anchors[frame] = G8L4_MS * fr["B"]["default_ms"] / fr["B_skinny"]["default_ms"]
    pred["B"]["ms_from_g8l4_anchor"] = anchors
    pred["B"]["ms_anchor_mean"] = sum(anchors.values()) / 2
    pred["B"]["gain_anchor_pct"] = (1 - pred["B"]["ms_anchor_mean"] / REF_MS) * 100
    # model error seen for g8l4 (predicted vs measured), for context
    pred["B_skinny"]["measured_ms"] = G8L4_MS
    pred["B_skinny"]["measured_gain_pct"] = (1 - G8L4_MS / REF_MS) * 100
    res["predictions"] = pred
    gate_gain = max(pred["A"]["gain_vs_ref_pct"], pred["B"]["gain_vs_ref_pct"], pred["B"]["gain_anchor_pct"])
    res["gate"] = {"best_gain_pct": gate_gain, "A_gain_pct": pred["A"]["gain_vs_ref_pct"], "B_gain_pct": pred["B"]["gain_vs_ref_pct"],
                   "B_gain_anchor_pct": pred["B"]["gain_anchor_pct"], "passes": gate_gain >= 8.0}
    # ratio to OLMo 2 int8 drafted
    res["ratio"] = {"olmo_ms": OLMO_I8_MS,
                    "A": pred["A"]["ms_mean"] / OLMO_I8_MS, "B": pred["B"]["ms_mean"] / OLMO_I8_MS,
                    "B_anchor": pred["B"]["ms_anchor_mean"] / OLMO_I8_MS, "ref": REF_MS / OLMO_I8_MS}
    (out_dir / "costmodel.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    md = ["# Final frontier: Stage 0 cost model", "",
          "## Local linears of a verify round (ms; 5 layers x 8 linears on K + 1 rows plus 2 projections of 1 row)", "",
          "| K | rows | bf16 cuBLAS | bf16 skinny | int8 skinny | int4 tinygemm | int4 skinny | int4 dispatch |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ks:
        md.append(f"| {k} | {k + 1} | " + " | ".join(f"{local_ms[l][k]:.3f}" for l in ("bf16", "bf16s", "int8s", "int4", "int4s", "int4d")) + " |")
    md += ["", "First row count where the skinny int4 kernel beats tinygemm: " + ", ".join(f"{k}: {v[0] if v else 'never (<= 33)'}" for k, v in res["crossover_first_m_where_skinny_wins"].items()), ""]
    for frame in ("bf16", "int4"):
        fr = res["frames"][frame]
        md += [f"## Model, {frame} frame (bf16 pieces + bf16 tallies / int4 pieces + int4 tallies)", "",
               "| variant | k_cross / k_within | verify pair K=0 / 6 (ms) | B(P=1) ms | bytes/global | rounds/byte | ms/byte ext | ms/byte default | vs ref |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for key in ("ref", "A", "B_skinny", "B_tiny", "B"):
            r = fr[key]
            md.append(f"| {r['name']} | {r['k_cross']} / {r['k_within']} | {r['pair_ms']['0']:.3f} / {r['pair_ms']['6']:.3f} | {r['B_ms'][1]:.3f} | {r['bytes_per_global']:.1f} | "
                      f"{r['rounds_per_byte']:.3f} | {r['ext_ms']:.3f} | {r['default_ms']:.3f} | {(1 - r['default_ms'] / fr['ref']['default_ms']) * 100:+.1f}% |")
        md.append("")
    md += ["## Predictions applied to the measured reference (default prompts, 256 new bytes)", "",
           f"Reference: global int8 + bf16 local, P4, measured {REF_MS} ms/byte. g8l4 (skinny int4 local) measured {G8L4_MS}.", "",
           "| configuration | model ratio to ref (bf16 frame / int4 frame) | predicted ms/byte (mean of frames) | gain vs reference |", "|---|---|---:|---:|"]
    for key, label in (("A", "A: local int8 + global int8"), ("B_skinny", "B': local int4 skinny (measured g8l4)"), ("B_tiny", "B'': local int4 tinygemm"), ("B", "B: local int4 dispatch")):
        p = pred[key]
        md.append(f"| {label} | {p['model_ratio_to_ref']['bf16']:.3f} / {p['model_ratio_to_ref']['int4']:.3f} | {p['ms_mean']:.3f} | {p['gain_vs_ref_pct']:+.1f}% |")
    md += ["", f"B anchored on the measured g8l4 (x model(B)/model(B') per frame): {anchors['bf16']:.3f} / {anchors['int4']:.3f}, mean {pred['B']['ms_anchor_mean']:.3f} ms/byte = {pred['B']['gain_anchor_pct']:+.1f}% vs reference.", "",
           f"**Gate (H65): best predicted gain {gate_gain:+.1f}% (A {pred['A']['gain_vs_ref_pct']:+.1f}%, B {pred['B']['gain_vs_ref_pct']:+.1f}%, B anchored {pred['B']['gain_anchor_pct']:+.1f}%) vs 8% -> {'PASS' if res['gate']['passes'] else 'STOP'}**", "",
           f"Ratio to OLMo 2 int8 drafted ({OLMO_I8_MS} ms/byte, same session as the reference): reference {res['ratio']['ref']:.2f}x, A {res['ratio']['A']:.2f}x, B {res['ratio']['B']:.2f}x (anchored {res['ratio']['B_anchor']:.2f}x).", ""]
    (out_dir / "costmodel.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
