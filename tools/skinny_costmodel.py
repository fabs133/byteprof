"""Stages 0/1 of TASK_skinny_kernels.md: per-pass kernel sums and the P4 cost model with the Stage 1 kernel times.

Per-pass sums use the microbenchmark tables (results/skinny/stage0.json, stage1.json: us per call by shape and M, CUDA
graph, distinct weights per call) and the layer inventory of the models. The cost model applies them as DELTAS to the
measured pieces of the existing P4 model (tools/p4_costmodel.py): a piece's new time = measured time
 - sum over its linears of the baseline kernel time + sum over its linears of the new kernel time, at the piece's row
count (a local verify round at K drafted bytes runs its 5 layers' linears on K + 1 rows; a global pass at P patches on
P rows). The tiny 16-row gate projections, attention, norms, mLSTM recurrences and everything else stay as measured.
Acceptance tallies are the bf16 / int4 P4 tallies (results/p4/tallies_*.json); a different weight precision changes the
generated text a little (as measured before), which is not modelled.

Usage: python tools/skinny_costmodel.py [--stage1 results/skinny/stage1.json] [--out results/skinny]
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

# (out, in, count) per layer
LOCAL_LAYER = [(1024, 2048, 2), (2048, 2048, 3), (2816, 2048, 2), (2048, 2816, 1)]
GLOBAL_LAYER = [(2048, 2048, 4), (8192, 2048, 2), (2048, 8192, 1)]
N_LOCAL_LAYERS = 5  # 1 encoder + 4 decoder layers run in every verify round
N_GLOBAL_LAYERS = 16  # Bolmo's global transformer and OLMo 2's blocks
BACKENDS = {  # precision label -> (stage 0/1 backend)
    "bf16": "bf16_cublas", "int8": "int8_cur", "int4": "int4_tiny",
    "bf16s": "bf16_skinny", "int8s": "int8_skinny", "int4s": "int4_skinny",
}
MS_GRID = (1, 2, 3, 4, 5, 8, 9, 16, 17, 33)
# measured P4 ms/byte on the default prompts (FINDINGS.md, Boundary-gated drafting): the model predicts 2.213 / 1.458
MEASURED_REF = {"bf16": 1.934, "int4": 1.456}


class Table:
    def __init__(self, *jsons: dict):
        self.cells: dict = {}
        for j in jsons:
            self.cells.update(j["cells"])
        self.copy_gbs = jsons[0]["copy_gbs"]

    def us(self, label: str, n: int, k: int, m: int) -> float:
        """Linear interpolation in M between the measured rows (clamped to 1..33)."""
        c = self.cells[f"{BACKENDS[label]}|{n}x{k}"]["m"]
        xs = sorted(int(x) for x in c)
        m = min(max(m, xs[0]), xs[-1])
        for a, b in zip(xs, xs[1:]):
            if a <= m <= b:
                ya, yb = c[str(a)]["us"], c[str(b)]["us"]
                return ya + (yb - ya) * (m - a) / (b - a)
        return c[str(xs[-1])]["us"]

    def layer_us(self, label: str, layer: list, m: int) -> float:
        return sum(cnt * self.us(label, n, k, m) for n, k, cnt in layer)


def local_pass_us(t: Table, label: str, rows: int) -> float:
    """Linears of one local verify round on `rows` bytes: 5 layers plus encoder out_projection and decoder
    in_projection (1 row each)."""
    return N_LOCAL_LAYERS * t.layer_us(label, LOCAL_LAYER, rows) + 2 * t.us(label, 2048, 2048, 1)


def global_pass_us(t: Table, label: str, rows: int) -> float:
    return 0.0 if rows == 0 else N_GLOBAL_LAYERS * t.layer_us(label, GLOBAL_LAYER, rows)


def sums_table(t: Table, labels: list[str]) -> dict:
    out: dict = {"global": {}, "local": {}, "olmo": {}}
    for lab in labels:
        out["global"][lab] = {p: global_pass_us(t, lab, p) for p in range(1, 9)}
        out["local"][lab] = {k: local_pass_us(t, lab, k + 1) for k in (0, 1, 2, 4, 6, 8, 12, 16, 24, 32)}
        out["olmo"][lab] = {n: global_pass_us(t, lab, n) for n in range(1, 10)}
    return out


class Variant:
    """A measured-baseline precision plus per-group kernel swaps."""

    def __init__(self, name: str, base_prec: str, local: tuple[str, str], glob: tuple[str, str], olmo_blocks: str | None = None):
        self.name, self.base_prec, self.local, self.glob, self.olmo = name, base_prec, local, glob, olmo_blocks


VARIANTS = [
    Variant("bf16 P4 (baseline, cuBLAS)", "bf16", ("bf16", "bf16"), ("bf16", "bf16")),
    Variant("bf16 P4, skinny bf16 local + global", "bf16", ("bf16", "bf16s"), ("bf16", "bf16s")),
    Variant("global int8 (skinny) + local bf16 (skinny)", "bf16", ("bf16", "bf16s"), ("bf16", "int8s")),
    Variant("global int8 (skinny) + local int4 (skinny)", "bf16", ("bf16", "int4s"), ("bf16", "int8s")),
    Variant("all-int4, tinygemm (today)", "int4", ("int4", "int4"), ("int4", "int4")),
    Variant("all-int4, skinny int4", "int4", ("int4", "int4s"), ("int4", "int4s")),
    Variant("global int8 (skinny) + local int4 tinygemm", "int4", ("int4", "int4"), ("int4", "int8s")),
]


def adjusted_inputs(prec: str, t: Table, v: Variant) -> acm.Inputs:
    I = acm.Inputs(prec)
    for k in list(I.pair):
        I.pair[k] += (local_pass_us(t, v.local[1], k + 1) - local_pass_us(t, v.local[0], k + 1)) / 1000
    for p in list(I.B):
        if p:
            I.B[p] += (global_pass_us(t, v.glob[1], p) - global_pass_us(t, v.glob[0], p)) / 1000
    return I


def best_cell(prec: str, I: acm.Inputs, ex: dict, retok_us: float) -> dict:
    p4t = json.loads((ROOT / f"results/p4/tallies_{prec}.json").read_text(encoding="utf-8"))["cells"]
    built = {}
    for key, tl in p4t.items():
        s, _, kc, kw = key.split("|")
        built[(s, int(kc), kw)] = p4.predict(tl, I, ex, retok_us)
    kcs = sorted({kc for _, kc, _ in built})
    kws = sorted({kw for _, _, kw in built})
    best = min(((kc, kw) for kc in kcs for kw in kws if ("ext", kc, kw) in built), key=lambda x: built[("ext", x[0], x[1])]["ms_per_byte"])
    return {"k_cross": best[0], "k_within": best[1], "ext": built[("ext", *best)], "default": built[("default", *best)],
            "all": {f"{s}|{kc}|{kw}": v for (s, kc, kw), v in built.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0", default="results/skinny/stage0.json")
    ap.add_argument("--stage1", default="results/skinny/stage1.json")
    ap.add_argument("--out", default="results/skinny")
    a = ap.parse_args()
    s0 = json.loads((ROOT / a.stage0).read_text(encoding="utf-8"))
    s1 = json.loads((ROOT / a.stage1).read_text(encoding="utf-8"))
    t = Table(s0, s1)
    out_dir = ROOT / a.out
    sums = sums_table(t, list(BACKENDS))
    res: dict = {"sums": sums, "variants": {}}
    md = ["# Skinny kernels: per-pass kernel sums and the P4 cost model", "",
          f"Device copy bandwidth {t.copy_gbs:.0f} GB/s (read + write).", "",
          "## Linear-kernel sums per pass (ms, microbenchmark tables)", "",
          "Global pass (16 layers x 7 linears) at P rows, and an OLMo 2 verify pass at T tokens (same blocks, no lm_head):", "",
          "| rows | " + " | ".join(BACKENDS) + " |", "|---|" + "---:|" * len(BACKENDS)]
    for p in range(1, 9):
        md.append(f"| {p} | " + " | ".join(f"{sums['global'][lab][p] / 1000:.2f}" for lab in BACKENDS) + " |")
    md += ["", "Local verify round (5 layers x 8 linears on K + 1 rows, plus 2 projections of 1 row):", "",
           "| K | " + " | ".join(BACKENDS) + " |", "|---|" + "---:|" * len(BACKENDS)]
    for k in sums["local"]["bf16"]:
        md.append(f"| {k} | " + " | ".join(f"{sums['local'][lab][k] / 1000:.3f}" for lab in BACKENDS) + " |")
    md.append("")
    retok = acm.RETOK_US
    for prec in ("bf16", "int4"):
        ex = p4.load_extra(prec)
        for v in (x for x in VARIANTS if x.base_prec == prec):
            I = adjusted_inputs(prec, t, v)
            res["variants"][v.name] = {"base_prec": prec, **best_cell(prec, I, ex, retok),
                                       "pair": I.pair, "B": I.B}
    base = res["variants"][VARIANTS[0].name]
    base4 = res["variants"]["all-int4, tinygemm (today)"]
    md += ["## P4 cost-model predictions (ms/byte, best (k_cross, k_within) on the extended prompt set)", "",
           "| variant | k_cross / k_within | B/global ext | ms/byte ext | ms/byte default | vs reference (default) | scaled to the measured reference |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        r = res["variants"][v.name]
        ref = base if v.base_prec == "bf16" else base4
        md.append(f"| {v.name} | {r['k_cross']} / {r['k_within']} | {r['ext']['bytes_per_global']:.1f} | {r['ext']['ms_per_byte']:.3f} | "
                  f"{r['default']['ms_per_byte']:.3f} | {(1 - r['default']['ms_per_byte'] / ref['default']['ms_per_byte']) * 100:+.1f}% | "
                  f"{MEASURED_REF[v.base_prec] * r['default']['ms_per_byte'] / ref['default']['ms_per_byte']:.3f} |")
    # gate
    nl = res["variants"]["global int8 (skinny) + local bf16 (skinny)"]["default"]["ms_per_byte"]
    g1 = (1 - nl / base["default"]["ms_per_byte"]) * 100
    ai4 = res["variants"]["all-int4, skinny int4"]["default"]["ms_per_byte"]
    g2 = (1 - ai4 / base4["default"]["ms_per_byte"]) * 100
    res["gate"] = {"near_lossless_gain_pct": g1, "all_int4_gain_pct": g2, "passes": g1 >= 15.0 or g2 >= 8.0,
                   "near_lossless_passes": g1 >= 15.0, "all_int4_passes": g2 >= 8.0}
    md += ["", f"**Gate (default prompts): near-lossless (global int8 + bf16 local) {g1:+.1f}% vs bf16 P4 (needs >= +15%); "
               f"all-int4 skinny {g2:+.1f}% vs tinygemm all-int4 P4 (needs >= +8%) -> "
               f"{'PASS' if res['gate']['passes'] else 'STOP'}**", ""]
    # OLMo 2: blocks int8 skinny vs bf16 cuBLAS (the verify pass is the same 16 layers, T = k + 1 tokens)
    olmo = {}
    for k in (0, 1, 2, 4, 8):
        old = acm.olmo_verify(k)
        new = old + (global_pass_us(t, "int8s", k + 1) - global_pass_us(t, "bf16", k + 1)) / 1000
        olmo[k] = {"old_ms": old, "new_ms": new}
    res["olmo_verify"] = olmo
    md += ["## OLMo 2 verify pass with int8 skinny blocks (ms; lm_head, embeddings and attention unchanged)", "",
           "| drafted tokens k | bf16 (measured) | int8 skinny (delta model) |", "|---|---:|---:|"]
    for k, v in olmo.items():
        md.append(f"| {k} | {v['old_ms']:.2f} | {v['new_ms']:.2f} |")
    ratio_old = p4.OLMO_LOOKUP
    k2 = 2  # K = 8 bytes is about 2 tokens (acm.K_TOK)
    ol_new = ratio_old * olmo[k2]["new_ms"] / olmo[k2]["old_ms"]
    res["olmo_pred_ms_per_byte"] = ol_new
    bolmo_i8 = res["variants"]["global int8 (skinny) + local bf16 (skinny)"]["default"]["ms_per_byte"]
    res["ratio_pred"] = bolmo_i8 / ol_new
    md += ["", f"OLMo 2 drafted (K = 8) with int8 skinny blocks: {ratio_old:.3f} x {olmo[k2]['new_ms']:.2f}/{olmo[k2]['old_ms']:.2f} = "
               f"**{ol_new:.3f} ms/byte** (assumes the pass time scales as the verify pass at 3 tokens); Bolmo/OLMo 2 predicted "
               f"{bolmo_i8:.3f} / {ol_new:.3f} = **{res['ratio_pred']:.2f}x**.", ""]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "costmodel.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (out_dir / "costmodel.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
