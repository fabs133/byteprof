"""Stage A0 / A2 tables for FINDINGS.md from results/parallel: round cost vs K for both paths (figure F6 data), marginal
cost per extra position, the K = 0 -> 1 jump, and the kernel-class growth from the nsys traces.
-> results/parallel/tables.md"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "results" / "parallel"
CLASSES = ["mlstm_step", "commit", "linear", "lm_head", "elementwise", "scan", "other"]


def bench(tag: str) -> dict[int, float]:
    d = json.loads((P / "bench" / tag / "bench.json").read_text(encoding="utf-8"))
    return {int(k): v["ms"] for k, v in d["rounds"].items()}


def classes(*tags: str) -> dict[int, dict]:
    out: dict = {}
    for t in tags:
        out.update({int(k): v for k, v in json.loads((P / "nsys" / f"{t}-classes.json").read_text()).items()})
    return out


def main() -> int:
    md: list[str] = []
    runs = {"bf16": {"steps": "bf16-steps", "fused": "bf16-fused-bv16", "fused (L = 0)": "bf16-fused-L0"},
            "int4": {"steps": "int4-steps", "fused": "int4-fused-bv16", "fused (L = 0)": "int4-fused-L0"}}
    md += ["## Round cost vs K (GPU ms per replay of A_K + C_K, median of 5 trials of 500; no new patch)", ""]
    for prec, tags in runs.items():
        data = {n: bench(t) for n, t in tags.items()}
        ks = sorted(data["steps"])
        md += [f"### {prec}", "", "| K | " + " | ".join(f"{n} ms | {n} / K=0" for n in tags) + " |",
               "|---:|" + "---:|---:|" * len(tags)]
        for k in ks:
            md.append(f"| {k} | " + " | ".join(f"{data[n][k]:.4f} | {data[n][k] / data[n][0]:.3f}" for n in tags) + " |")
        md += ["", "| path | K=0 -> 1 jump (ms) | us per extra position K=1 -> 8 | K=1 -> 16 | K=1 -> 32 | K=16 / K=0 | K=32 / K=0 |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for n in tags:
            d = data[n]
            md.append(f"| {n} | {d[1] - d[0]:.3f} | {(d[8] - d[1]) / 7 * 1000:.1f} | {(d[16] - d[1]) / 15 * 1000:.1f} | "
                      f"{(d[32] - d[1]) / 31 * 1000:.1f} | {d[16] / d[0]:.3f} | {d[32] / d[0]:.3f} |")
        md.append("")
    md += ["## Kernel classes from the nsys traces (us per round, 30 back-to-back rounds per K, count in brackets)", ""]
    traces = {"bf16 steps": ("bf16",), "bf16 fused": ("bf16-fused-a", "bf16-fused-b"), "int4 steps": ("int4",),
              "int4 fused": ("int4-fused-a",)}
    for name, tags in traces.items():
        c = classes(*tags)
        ks = sorted(c)
        md += [f"### {name}", "", "| class | " + " | ".join(f"K={k}" for k in ks) + " |", "|---|" + "---:|" * len(ks)]
        for cl in CLASSES:
            md.append(f"| {cl} | " + " | ".join(f"{c[k]['classes'][cl]['us']:.1f} ({c[k]['classes'][cl]['count']:.0f})" for k in ks) + " |")
        md.append("| kernel sum | " + " | ".join(f"{c[k]['kernel_sum_us']:.1f} ({c[k]['kernels']})" for k in ks) + " |")
        md.append("| first-to-last span | " + " | ".join(f"{c[k]['span_us']:.1f}" for k in ks) + " |")
        md.append("")
        if 1 in c and 8 in c:
            tot = c[8]["kernel_sum_us"] - c[1]["kernel_sum_us"]
            span = c[8]["span_us"] - c[1]["span_us"]
            md += [f"Growth K=1 -> 8: kernel sum {tot:+.1f} us, span {span:+.1f} us (gaps {span - tot:+.1f}); per class: " +
                   ", ".join(f"{cl} {c[8]['classes'][cl]['us'] - c[1]['classes'][cl]['us']:+.1f} "
                             f"({(c[8]['classes'][cl]['us'] - c[1]['classes'][cl]['us']) / tot * 100:.0f}%)" for cl in CLASSES), ""]
        if 0 in c and 1 in c:
            tot = c[1]["span_us"] - c[0]["span_us"]
            md += [f"Jump K=0 -> 1: span {tot:+.1f} us (kernel sum {c[1]['kernel_sum_us'] - c[0]['kernel_sum_us']:+.1f}, kernels "
                   f"{c[0]['kernels']} -> {c[1]['kernels']}); per class: " +
                   ", ".join(f"{cl} {c[1]['classes'][cl]['us'] - c[0]['classes'][cl]['us']:+.1f}" for cl in CLASSES), ""]
    (P / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
