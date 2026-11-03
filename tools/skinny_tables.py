"""Kernel tables of TASK_skinny_kernels.md (Stage 0 vs Stage 1) and the H54 / H55 checks from results/skinny/stage{0,1}.json.
Output: results/skinny/tables.md, results/skinny/h54_h55.json.  Usage: python tools/skinny_tables.py"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
s0 = json.loads((ROOT / "results/skinny/stage0.json").read_text(encoding="utf-8"))
s1 = json.loads((ROOT / "results/skinny/stage1.json").read_text(encoding="utf-8"))
cells = {**s0["cells"], **s1["cells"]}
copy = s0["copy_gbs"]
SHAPES = [(2048, 2048), (8192, 2048), (2048, 8192), (1024, 2048), (2816, 2048), (2048, 2816)]
MS = (1, 2, 3, 4, 5, 8, 9, 16, 17, 33)
PAIRS = [("bf16_cublas", "bf16_skinny"), ("int8_cur", "int8_skinny"), ("int4_tiny", "int4_skinny")]
NAMES = {"bf16_cublas": "bf16 cuBLAS", "bf16_skinny": "bf16 skinny", "int8_cur": "int8 today (GEMV at M=1, dequantize + cuBLAS for M>1)",
         "int8_skinny": "int8 skinny", "int4_tiny": "int4 tinygemm", "int4_skinny": "int4 skinny"}


def c(be, n, k):
    return cells[f"{be}|{n}x{k}"]


md = ["# Skinny kernels: Stage 0 (today's kernels) vs Stage 1 (skinny), us per call in a CUDA graph (ring of distinct weights)", "",
      f"Device copy bandwidth {copy:.0f} GB/s (256 MiB device-to-device copy, read + write counted; the same figure is the floor reference "
      f"for weight bytes / time). GB/s = stored weight bytes (incl. scales) / time. Floor = stored bytes / copy bandwidth.", ""]
for be0, be1 in PAIRS:
    md += [f"## {NAMES[be0]} vs {NAMES[be1]}", "", "us per call (GB/s in brackets for M = 1 and M = 9):", "",
           "| shape (out x in) | floor us | backend | " + " | ".join(f"M={m}" for m in MS) + " |", "|---|---:|---|" + "---:|" * len(MS)]
    for n, k in SHAPES:
        for be in (be0, be1):
            cc = c(be, n, k)
            row = []
            for m in MS:
                v = cc["m"][str(m)]
                row.append(f"{v['us']:.1f}" + (f" ({v['gbs']:.0f})" if m in (1, 9) else ""))
            md.append(f"| {n} x {k} | {cc['m']['1']['floor_us']:.1f} | {NAMES[be].split(' (')[0]} | " + " | ".join(row) + " |")
    md.append("")
# H54: int8 skinny on the global shapes
h54 = {}
for n, k in [(2048, 2048), (8192, 2048), (2048, 8192)]:
    cc = c("int8_skinny", n, k)["m"]
    h54[f"{n}x{k}"] = {"gbs_m1": cc["1"]["gbs"], "frac_of_copy": cc["1"]["gbs"] / copy, "us_m1": cc["1"]["us"], "us_m8": cc["8"]["us"],
                       "ratio_m8_m1": cc["8"]["us"] / cc["1"]["us"], "ratio_max_1_to_9": max(cc[str(m)]["us"] for m in (2, 3, 4, 5, 8, 9)) / cc["1"]["us"]}
# H55: bf16 skinny at M = 2..9 against cuBLAS M = 1 on the local shapes
h55 = {}
for n, k in [(2048, 2048), (1024, 2048), (2816, 2048), (2048, 2816)]:
    base = c("bf16_cublas", n, k)["m"]["1"]["us"]
    sk = c("bf16_skinny", n, k)["m"]
    worst = max(sk[str(m)]["us"] for m in (2, 3, 4, 5, 8, 9))
    h55[f"{n}x{k}"] = {"cublas_m1_us": base, "skinny_worst_2_9_us": worst, "ratio": worst / base}
md += ["## H54 (int8 efficiency on the global shapes)", "", "| shape | us at M=1 | GB/s | % of copy bandwidth | us at M=8 | M=8 / M=1 | worst M=2..9 / M=1 |", "|---|---:|---:|---:|---:|---:|---:|"]
for s, v in h54.items():
    md.append(f"| {s} | {v['us_m1']:.1f} | {v['gbs_m1']:.0f} | {v['frac_of_copy'] * 100:.0f}% | {v['us_m8']:.1f} | {v['ratio_m8_m1']:.2f} | {v['ratio_max_1_to_9']:.2f} |")
md += ["", "## H55 (bf16 skinny at M = 2..9 against cuBLAS at M = 1, local shapes)", "", "| shape | cuBLAS M=1 us | skinny worst M=2..9 us | ratio |", "|---|---:|---:|---:|"]
for s, v in h55.items():
    md.append(f"| {s} | {v['cublas_m1_us']:.1f} | {v['skinny_worst_2_9_us']:.1f} | {v['ratio']:.2f} |")
(ROOT / "results/skinny/tables.md").write_text("\n".join(md), encoding="utf-8")
(ROOT / "results/skinny/h54_h55.json").write_text(json.dumps({"copy_gbs": copy, "h54": h54, "h55": h55}, indent=1), encoding="utf-8")
print("\n".join(md[-22:]))
