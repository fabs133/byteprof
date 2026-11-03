"""Stage 2 speed tables of TASK_skinny_kernels.md from results/skinny/speed/<batch>/<cell>-run<r>.json.

Per cell: ms/byte of both runs, their difference, mean, rounds/byte, bytes per global pass. Per comparison: the full-prompt
change and the change on the rows whose generated text is identical in both cells (rows are matched by position: same
prompt, same run). A change is the reduction of ms/byte, (1 - new / reference) * 100.
Output: results/skinny/speed_report.{md,json}.  Usage: python tools/skinny_speed_report.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEED = ROOT / "results/skinny/speed"

# (batch, cell file stem) per label; references first
CELLS = {
    "B1": ["bf16-none", "bf16-p4", "bolmo-g8-none", "bolmo-g8-p4"],
    "B2": ["bolmo-bf16sk-none", "bolmo-bf16sk-p4", "bolmo-g8l4-none", "bolmo-g8l4-p4"],
    "B3": ["int4-none", "int4-p4", "bolmo-all4-none", "bolmo-all4-p4"],
    "B4": ["bolmo-g8-none", "bolmo-g8-p4", "olmo-int8sk-none", "olmo-int8sk-k8", "olmo-bf16-none", "olmo-bf16-combined-k8"],
}
# (label, batch, new, reference)
COMPARE = [
    ("H56: g8 P4 vs bf16 P4", "B1", "bolmo-g8-p4", "bf16-p4"),
    ("g8 minimal vs bf16 minimal", "B1", "bolmo-g8-none", "bf16-none"),
    ("bf16 skinny P4 vs bf16 P4 (kernel only)", "B2", "bolmo-bf16sk-p4", ("B1", "bf16-p4")),
    ("bf16 skinny minimal vs bf16 minimal (kernel only)", "B2", "bolmo-bf16sk-none", ("B1", "bf16-none")),
    ("g8l4 P4 vs bf16 P4", "B2", "bolmo-g8l4-p4", ("B1", "bf16-p4")),
    ("g8l4 minimal vs bf16 minimal", "B2", "bolmo-g8l4-none", ("B1", "bf16-none")),
    ("H57: all-int4 skinny P4 vs all-int4 tinygemm P4", "B3", "bolmo-all4-p4", "int4-p4"),
    ("all-int4 skinny minimal vs tinygemm minimal", "B3", "bolmo-all4-none", "int4-none"),
    ("OLMo 2 int8 skinny K=8 vs OLMo 2 bf16 K=8", "B4", "olmo-int8sk-k8", "olmo-bf16-combined-k8"),
    ("OLMo 2 int8 skinny minimal vs bf16 minimal", "B4", "olmo-int8sk-none", "olmo-bf16-none"),
]


def load(batch: str, cell: str) -> list[dict]:
    out = []
    for r in (1, 2):
        f = SPEED / batch / f"{cell}-run{r}.json"
        if f.exists():
            out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


def subset(a: list[dict], b: list[dict]) -> tuple[float, float, int, int]:
    """ms/byte of cells a and b on the rows whose text is identical (same run, same position)."""
    ma = mb = ba = bb = n = tot = 0
    for ra, rb in zip(a, b):
        for x, y in zip(ra["rows"], rb["rows"]):
            tot += 1
            if x["text"] == y["text"]:
                n += 1
                ma += x["ms"]; ba += x["new_bytes"]; mb += y["ms"]; bb += y["new_bytes"]
    return (ma / ba if ba else float("nan"), mb / bb if bb else float("nan"), n, tot)


def main() -> int:
    md = ["# Skinny kernels: speed cells (default prompts, 256 new bytes for Bolmo / 72 tokens for OLMo 2, two runs per cell)", ""]
    res: dict = {"cells": {}, "compare": {}}
    for batch, cells in CELLS.items():
        md += [f"## Batch {batch}", "", "| cell | run 1 | run 2 | diff | mean ms/byte | rounds/byte | bytes/global pass |", "|---|---:|---:|---:|---:|---:|---:|"]
        for c in cells:
            runs = load(batch, c)
            if not runs:
                continue
            ms = [r["ms_per_byte"] for r in runs]
            em = sum(x.get("emitted", 0) for r in runs for x in r["rows"])
            gl = sum(x.get("global_calls", 0) for r in runs for x in r["rows"])
            diff = abs(ms[0] - ms[-1]) / min(ms) * 100 if len(ms) > 1 else float("nan")
            res["cells"][f"{batch}|{c}"] = {"ms": ms, "mean": sum(ms) / len(ms), "diff_pct": diff, "rounds_per_byte": runs[0].get("rounds_per_byte")}
            rpb = runs[0].get("rounds_per_byte")
            md.append(f"| {c} | {ms[0]:.3f} | {ms[-1]:.3f} | {diff:.1f}% | {sum(ms) / len(ms):.3f} | "
                      f"{'-' if rpb is None else f'{rpb:.3f}'} | {f'{em / gl:.1f}' if gl else '-'} |")
        md.append("")
    md += ["## Comparisons (reduction of ms/byte; identical-text rows only in the last columns)", "",
           "| comparison | new | reference | change (all rows) | identical rows | new / ref on them | change (identical rows) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, batch, new, ref in COMPARE:
        rb, rc = ref if isinstance(ref, tuple) else (batch, ref)
        a, b = load(batch, new), load(rb, rc)
        if not a or not b:
            continue
        ma = sum(r["ms_per_byte"] for r in a) / len(a)
        mb = sum(r["ms_per_byte"] for r in b) / len(b)
        sa, sb, n, tot = subset(a, b)
        res["compare"][label] = {"new": ma, "ref": mb, "change_pct": (1 - ma / mb) * 100, "identical": n, "rows": tot,
                                 "new_identical": sa, "ref_identical": sb, "change_identical_pct": (1 - sa / sb) * 100 if n else None}
        md.append(f"| {label} | {ma:.3f} | {mb:.3f} | {(1 - ma / mb) * 100:+.1f}% | {n}/{tot} | "
                  f"{(f'{sa:.3f} / {sb:.3f}' if n else '-')} | {(f'{(1 - sa / sb) * 100:+.1f}%' if n else '-')} |")
    # ratio block (H58)
    for tag, bol, ol in (("minimal", "bolmo-g8-none", "olmo-int8sk-none"), ("drafted", "bolmo-g8-p4", "olmo-int8sk-k8"),
                         ("bf16 minimal", None, None)):
        if bol is None:
            continue
        a, b = load("B4", bol), load("B4", ol)
        if a and b:
            ra = sum(r["ms_per_byte"] for r in a) / len(a)
            rb_ = sum(r["ms_per_byte"] for r in b) / len(b)
            res["compare"][f"ratio {tag} (B4)"] = {"bolmo": ra, "olmo": rb_, "ratio": ra / rb_}
            md += ["", f"Ratio block B4, {tag}: Bolmo {ra:.3f} / OLMo 2 {rb_:.3f} = **{ra / rb_:.3f}x**"]
    a, b = load("B4", "olmo-bf16-none"), load("B4", "olmo-bf16-combined-k8")
    if a and b:
        pass
    (ROOT / "results/skinny/speed_report.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (ROOT / "results/skinny/speed_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
