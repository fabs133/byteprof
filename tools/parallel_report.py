"""Stage A2 step 3 report: end-to-end cells (results/parallel/speed/*.json from tools/draft_speed.py), the identical-
output subset, and the cost model's predictions next to the measurements.  -> results/parallel/speed_report.{md,json}"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SP = ROOT / "results" / "parallel" / "speed"
CELLS = {"bf16": ["none", "steps-k6", "fused-k6", "fused-k8"], "int4": ["none", "steps-k4", "fused-k4", "fused-k6"]}
PRED_KEY = {"steps-k6": ("bf16", "steps", 6), "fused-k6": ("bf16", "fused", 6), "fused-k8": ("bf16", "fused", 8),
            "steps-k4": ("int4", "steps", 4), "fused-k4": ("int4", "fused", 4), "fused-k6": ("int4", "fused", 6)}


def load(prec: str, cell: str) -> list[dict]:
    return [json.loads((SP / f"{prec}-{cell}-run{r}.json").read_text(encoding="utf-8")) for r in (1, 2)]


def main() -> int:
    cm = json.loads((ROOT / "results/parallel/costmodel.json").read_text(encoding="utf-8"))
    md = ["# Stage A2: end-to-end (default prompts, 256 new bytes, 25 generations per run, two runs per cell)", ""]
    out: dict = {}
    for prec, cells in CELLS.items():
        runs = {c: load(prec, c) for c in cells}
        base = statistics.mean(r["ms_per_byte"] for r in runs["none"])
        steps_cell = cells[1]
        steps_ms = statistics.mean(r["ms_per_byte"] for r in runs[steps_cell])
        # identical-output prompts: the text of every (run, repeat) of the prompt equals the minimal loop's
        ref: dict[str, set[str]] = {}
        for r in runs["none"]:
            for row in r["rows"]:
                ref.setdefault(row["prompt"], set()).add(row["text"])
        same: dict[str, set[str]] = {}
        for c in cells[1:]:
            ok = {p for p in ref if all(row["text"] in ref[p] and len(ref[p]) == 1 for r in runs[c] for row in r["rows"]
                                        if row["prompt"] == p)}
            same[c] = ok
        common = set.intersection(*same.values())

        def ms_on(c: str, prompts: set[str]) -> float:
            rows = [row for r in runs[c] for row in r["rows"] if row["prompt"] in prompts]
            return sum(x["ms"] for x in rows) / sum(x["new_bytes"] for x in rows)

        md += [f"## {prec}", "", "| cell | run 1 | run 2 | pair gap | **mean ms/byte** | bytes/s | rounds/byte | bytes/round | "
               "vs minimal loop | vs `steps` | predicted gain vs `steps` |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        out[prec] = {}
        for c in cells:
            ms = [r["ms_per_byte"] for r in runs[c]]
            mean = statistics.mean(ms)
            rpb = runs[c][0].get("rounds_per_byte")
            pred = ""
            if c in PRED_KEY:
                _, v, k = PRED_KEY[c]
                p_steps = cm[prec]["steps"][f"default|{ {'bf16': 6, 'int4': 4}[prec] }"]["ms_per_byte"]
                p_this = cm[prec][v][f"default|{k}"]["ms_per_byte"]
                pred = f"{(1 - p_this / p_steps) * 100:+.1f}%" if c != steps_cell else "-"
            out[prec][c] = {"runs": ms, "mean": mean, "rounds_per_byte": rpb}
            md.append(f"| {c} | {ms[0]:.3f} | {ms[1]:.3f} | {abs(ms[0] - ms[1]) / min(ms) * 100:.1f}% | **{mean:.3f}** | "
                      f"{1000 / mean:.0f} | {rpb:.3f} | {(1 / rpb if rpb else 1):.2f} | {base / mean:.3f}x | {steps_ms / mean:.3f}x | {pred} |"
                      if rpb else f"| {c} | {ms[0]:.3f} | {ms[1]:.3f} | {abs(ms[0] - ms[1]) / min(ms) * 100:.1f}% | **{mean:.3f}** | "
                                  f"{1000 / mean:.0f} | - | 1.00 | 1.000x | {steps_ms / mean:.3f}x | |")
        md += ["", "Output identical to the minimal loop's (all 5 repeats, both runs): " +
               "; ".join(f"{c}: {len(same[c])}/5 prompts ({', '.join(sorted(same[c])) or '-'})" for c in cells[1:]),
               f"Prompts identical in every drafted cell: {len(common)} ({', '.join(sorted(common)) or '-'}).", ""]
        if common:
            md += ["Speed on those prompts only (ms/byte, both runs pooled):", "",
                   "| cell | ms/byte | vs minimal loop | vs `steps` |", "|---|---:|---:|---:|"]
            b, s_ = ms_on("none", common), ms_on(steps_cell, common)
            for c in cells:
                v = ms_on(c, common)
                md.append(f"| {c} | {v:.3f} | {b / v:.3f}x | {s_ / v:.3f}x |")
            md.append("")
    (ROOT / "results/parallel/speed_report.md").write_text("\n".join(md), encoding="utf-8")
    (ROOT / "results/parallel/speed_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
