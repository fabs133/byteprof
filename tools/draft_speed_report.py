"""Stage 3: speed table from results/drafting/speed/<group>/<cell>-run{1,2,...}.json.

Usage: python tools/draft_speed_report.py --groups bf16,int4 [--out results/drafting/stage3/speed_bolmo.md]
Per cell: runs, pair difference (rerun rule: > 5%), mean ms/byte, bytes/s, rounds per byte, emitted bytes per round,
speedup against the cell `none` of the same group, and (Bolmo) the Stage 1 prediction and the H25 ratio.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEED = ROOT / "results" / "drafting" / "speed"


def load_group(group: str) -> dict[str, list[dict]]:
    cells: dict[str, list[dict]] = {}
    for f in sorted((SPEED / group).glob("*-run*.json")):
        name, run = f.stem.rsplit("-run", 1)
        cells.setdefault(name, []).append({"run": int(run), **json.loads(f.read_text(encoding="utf-8"))})
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="bf16,int4")
    ap.add_argument("--out", default="results/drafting/stage3/speed_bolmo.md")
    ap.add_argument("--exclude", default="", help="comma list of group/cell-runN to leave out (discarded runs)")
    a = ap.parse_args()
    excl = set(x for x in a.exclude.split(",") if x)
    cm = None
    cmf = ROOT / "results/drafting/stage1/costmodel.json"
    if cmf.exists():
        cm = json.loads(cmf.read_text(encoding="utf-8"))
    out = []
    summary: dict = {}
    for group in a.groups.split(","):
        cells = load_group(group)
        out += [f"## {group}", "", "| cell | runs ms/byte | pair diff | **ms/byte** | bytes/s | rounds/byte | bytes/round | speedup vs none | predicted gain | measured gain | H25 ratio | peak MiB |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        none = None
        rows = {}
        for name, runs in cells.items():
            runs = [r for r in runs if f"{group}/{name}-run{r['run']}" not in excl]
            if not runs:
                continue
            vals = [r["ms_per_byte"] for r in runs]
            nb = sum(r["new_bytes"] for r in runs)
            rounds = sum(sum(x.get("rounds", 0) for x in r["rows"]) for r in runs)
            rows[name] = {"vals": vals, "mean": statistics.mean(vals), "diff": (max(vals) / min(vals) - 1) if len(vals) > 1 else 0.0,
                          "bps": statistics.mean(r["bytes_per_s"] for r in runs), "rpb": rounds / nb if rounds else None,
                          "bpr": nb / rounds if rounds else None, "peak": max(r["peak_mib"] for r in runs)}
        none = rows.get("none", {}).get("mean")
        for name, r in rows.items():
            kind, _, k = name.partition("-k")
            pred = meas = ratio = None
            if cm and name != "none" and group in cm:
                key = f"{'oracle (upper bound)' if kind == 'oracle' else kind}|{k}|default"
                p = cm[group]["predictions"].get(key)
                if p:
                    pred = p["gain_vs_k0_pred"]
                    meas = 1 - r["mean"] / none if none else None
                    ratio = meas / pred if (meas is not None and pred) else None
            out.append(f"| {name} | {', '.join(f'{v:.3f}' for v in r['vals'])} | {r['diff'] * 100:.1f}% | **{r['mean']:.3f}** | {r['bps']:.0f} | "
                       f"{r['rpb']:.3f} | {r['bpr']:.2f} | " if r["rpb"] else f"| {name} | {', '.join(f'{v:.3f}' for v in r['vals'])} | {r['diff'] * 100:.1f}% | **{r['mean']:.3f}** | {r['bps']:.0f} | - | - | ")
            out[-1] += (f"{none / r['mean']:.3f}x | " if none else "- | ")
            out[-1] += (f"{pred * 100:+.1f}% | {meas * 100:+.1f}% | {ratio:.2f} | " if pred is not None and meas is not None else "- | - | - | ")
            out[-1] += f"{r['peak']:.0f} |"
        out.append("")
        # ---- per prompt: ms/byte, bytes/round, and whether the generated text equals the minimal loop's ----
        def per_prompt(runs):
            d: dict[str, dict] = {}
            for r in runs:
                for row in r["rows"]:
                    e = d.setdefault(row["prompt"], {"ms": 0.0, "nb": 0, "rounds": 0, "text": row["text"]})
                    e["ms"] += row["ms"]
                    e["nb"] += row["new_bytes"]
                    e["rounds"] += row.get("rounds", 0)
            return d

        pp = {name: per_prompt([r for r in runs if f"{group}/{name}-run{r['run']}" not in excl]) for name, runs in cells.items()}
        if "none" in pp:
            prompts = list(pp["none"])
            out += [f"### {group}: per prompt (mean over runs), ms/byte | bytes per round | output identical to the minimal loop", "",
                    "| cell | " + " | ".join(prompts) + " | speedup on the prompts with identical output |", "|---|" + "---|" * (len(prompts) + 1)]
            for name, d in pp.items():
                cellsx, same = [], []
                for pr in prompts:
                    e = d.get(pr)
                    if not e:
                        cellsx.append("-")
                        continue
                    ident = e["text"] == pp["none"][pr]["text"]
                    if ident:
                        same.append(pr)
                    br = f"{e['nb'] / e['rounds']:.2f}" if e["rounds"] else "-"
                    cellsx.append(f"{e['ms'] / e['nb']:.3f} / {br} / {'=' if ident else 'differs'}")
                sub = ""
                if name != "none" and same:
                    t_cell = sum(d[x]["ms"] for x in same) / sum(d[x]["nb"] for x in same)
                    t_none = sum(pp["none"][x]["ms"] for x in same) / sum(pp["none"][x]["nb"] for x in same)
                    sub = f"{t_none / t_cell:.3f}x ({len(same)}/{len(prompts)} prompts)"
                    summary.setdefault(group, {}).setdefault(name, {})["speedup_identical"] = t_none / t_cell
                out.append(f"| {name} | " + " | ".join(cellsx) + f" | {sub or '-'} |")
            out.append("")
        summary.setdefault(group, {})
        for k, v in rows.items():
            summary[group].setdefault(k, {}).update(v)
    text = "\n".join(out)
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    (p.with_suffix(".json")).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
