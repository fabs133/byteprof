"""H53 (TASK_build_p4.md): same-session ratio block, Bolmo minimal / P4 against OLMo 2 minimal / combined lookup K = 8 (bf16).
-> results/p4/h53_report.md"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SP = ROOT / "results/p4/speed/h53"
CELLS = ["bolmo-bf16-none", "bf16-p4", "olmo-bf16-none", "olmo-bf16-combined-k8"]


def main() -> int:
    runs = {c: [json.loads((SP / f"{c}-run{r}.json").read_text(encoding="utf-8")) for r in (1, 2)] for c in CELLS}
    mean = {c: statistics.mean(r["ms_per_byte"] for r in runs[c]) for c in CELLS}
    md = ["# H53: same-session ratio block (bf16, default prompts; run 1 in the listed order, run 2 reversed)", "",
          "| cell | run 1 | run 2 | pair gap | mean ms/byte | bytes/s |", "|---|---:|---:|---:|---:|---:|"]
    for c in CELLS:
        ms = [r["ms_per_byte"] for r in runs[c]]
        md.append(f"| {c} | {ms[0]:.3f} | {ms[1]:.3f} | {abs(ms[0] - ms[1]) / min(ms) * 100:.1f}% | **{mean[c]:.3f}** | {1000 / mean[c]:.0f} |")
    per_run = [runs["bf16-p4"][i]["ms_per_byte"] / runs["olmo-bf16-combined-k8"][i]["ms_per_byte"] for i in (0, 1)]
    md += ["", f"Bolmo P4 / OLMo 2 combined lookup K = 8: **{mean['bf16-p4'] / mean['olmo-bf16-combined-k8']:.3f}x** (run 1 {per_run[0]:.3f}x, run 2 {per_run[1]:.3f}x).",
           f"Bolmo minimal / OLMo 2 minimal: {mean['bolmo-bf16-none'] / mean['olmo-bf16-none']:.3f}x.",
           f"Drafting gain: Bolmo {mean['bolmo-bf16-none'] / mean['bf16-p4']:.2f}x, OLMo 2 {mean['olmo-bf16-none'] / mean['olmo-bf16-combined-k8']:.2f}x."]
    (ROOT / "results/p4/h53_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
