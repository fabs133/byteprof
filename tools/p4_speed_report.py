"""Stage 2 report (TASK_build_p4.md): end-to-end cells from results/p4/speed/*.json (tools/draft_speed.py), the identical-output
subsets, the acceptance statistics of the gated loop and the cost model's predictions next to the measurements.
-> results/p4/speed_report.{md,json}"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SP = ROOT / "results" / "p4" / "speed"
CELLS = {"bf16": ["none", "p0-k6", "p0-k8", "p4"], "int4": ["none", "p0-k4", "p0-k6", "p4", "p4-kc12"]}
P0_BEST = {"bf16": "p0-k8", "int4": "p0-k4"}


def load(prec: str, cell: str) -> list[dict]:
    return [json.loads((SP / f"{prec}-{cell}-run{r}.json").read_text(encoding="utf-8")) for r in (1, 2)]


def main() -> int:
    cm = json.loads((ROOT / "results/p4/costmodel.json").read_text(encoding="utf-8"))
    md = ["# P4 end-to-end (default prompts, 256 new bytes, 25 generations per run, two runs per cell)", ""]
    out: dict = {}
    for prec, cells in CELLS.items():
        runs = {c: load(prec, c) for c in cells if (SP / f"{prec}-{c}-run1.json").exists()}
        cells = [c for c in cells if c in runs]
        base = statistics.mean(r["ms_per_byte"] for r in runs["none"])
        p0c = P0_BEST[prec]
        p0_ms = statistics.mean(r["ms_per_byte"] for r in runs[p0c])
        ref: dict[str, set[str]] = {}
        for r in runs["none"]:
            for row in r["rows"]:
                ref.setdefault(row["prompt"], set()).add(row["text"])
        same = {c: {p for p in ref if all(row["text"] in ref[p] and len(ref[p]) == 1 for r in runs[c] for row in r["rows"]
                                          if row["prompt"] == p)} for c in cells[1:]}
        main_cells = [c for c in cells[1:] if c in (p0c, "p4")]
        common = set.intersection(*[same[c] for c in main_cells])

        def ms_on(c: str, prompts: set[str]) -> float:
            rows = [row for r in runs[c] for row in r["rows"] if row["prompt"] in prompts]
            return sum(x["ms"] for x in rows) / sum(x["new_bytes"] for x in rows)

        pred_p4 = cm[prec]["P4 as built"]["default"]["ms_per_byte"]
        pred_p0 = cm[prec]["P0"]["default"]["ms_per_byte"]
        md += [f"## {prec}", "", "| cell | run 1 | run 2 | pair gap | **mean ms/byte** | bytes/s | rounds/byte | bytes/round | bytes/global pass | crossing rounds | vs minimal loop | vs best P0 |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        out[prec] = {}
        for c in cells:
            ms = [r["ms_per_byte"] for r in runs[c]]
            mean = statistics.mean(ms)
            rows = [row for r in runs[c] for row in r["rows"]]
            nb = sum(x["new_bytes"] for x in rows)
            rounds = sum(x.get("rounds", 0) for x in rows)
            glob = sum(x.get("global_calls", 0) for x in rows)
            kinds = {k: sum(x.get("kinds", {}).get(k, 0) for x in rows) for k in ("within", "cross")}
            cross = f"{kinds['cross'] / max(rounds, 1) * 100:.0f}%" if rounds and kinds["cross"] + kinds["within"] else "-"
            out[prec][c] = {"runs": ms, "mean": mean, "bytes_per_round": nb / rounds if rounds else 1.0,
                            "bytes_per_global": nb / glob if glob else None, "cross_share": kinds["cross"] / rounds if rounds else None}
            md.append(f"| {c} | {ms[0]:.3f} | {ms[1]:.3f} | {abs(ms[0] - ms[1]) / min(ms) * 100:.1f}% | **{mean:.3f}** | {1000 / mean:.0f} | "
                      + (f"{rounds / nb:.3f} | {nb / rounds:.2f} | {nb / glob:.1f} | {cross} |" if rounds else "- | 1.00 | - | - |")
                      + f" {base / mean:.3f}x | {p0_ms / mean:.3f}x |")
        md += ["", "Output identical to the minimal loop's (all 5 repeats, both runs): " +
               "; ".join(f"{c}: {len(same[c])}/5 prompts ({', '.join(sorted(same[c])) or '-'})" for c in cells[1:]),
               f"Prompts identical in {' and '.join(main_cells)}: {len(common)} ({', '.join(sorted(common)) or '-'}).", ""]
        if common:
            md += ["Speed on those prompts only (ms/byte, both runs pooled):", "", "| cell | ms/byte | vs minimal loop | vs best P0 |", "|---|---:|---:|---:|"]
            b, s_ = ms_on("none", common), ms_on(p0c, common)
            for c in cells:
                v = ms_on(c, common)
                md.append(f"| {c} | {v:.3f} | {b / v:.3f}x | {s_ / v:.3f}x |")
            out[prec]["identical_subset"] = {c: ms_on(c, common) for c in cells} | {"prompts": sorted(common)}
            md.append("")
        # P4 against P0 on prompts where the two loops emit the same text as each other (whatever the minimal loop does)
        t0 = {(row["prompt"]): {r2["text"] for r in runs[p0c] for r2 in r["rows"] if r2["prompt"] == row["prompt"]} for row in runs[p0c][0]["rows"]}
        t4 = {(row["prompt"]): {r2["text"] for r in runs["p4"] for r2 in r["rows"] if r2["prompt"] == row["prompt"]} for row in runs["p4"][0]["rows"]}
        agree = {p for p in t0 if t0[p] == t4[p] and len(t0[p]) == 1}
        if agree:
            v0, v4 = ms_on(p0c, agree), ms_on("p4", agree)
            md += [f"Prompts on which P0 ({p0c}) and P4 emit the same text: {len(agree)} ({', '.join(sorted(agree))}): "
                   f"P0 {v0:.3f}, P4 {v4:.3f} ms/byte, P4 {(1 - v4 / v0) * 100:+.1f}% faster.", ""]
            out[prec]["p0_p4_same_text"] = {"prompts": sorted(agree), "p0": v0, "p4": v4}
        # prediction vs measurement
        meas = out[prec]["p4"]["mean"]
        md += [f"P4 predicted (Stage 0 cost model, default prompts, chosen on the extended set) {pred_p4:.3f} ms/byte; measured {meas:.3f} "
               f"({(meas / pred_p4 - 1) * 100:+.1f}%). P0 predicted {pred_p0:.3f}, measured {p0_ms:.3f} ({(p0_ms / pred_p0 - 1) * 100:+.1f}%). "
               f"P4 vs P0 measured {(1 - meas / p0_ms) * 100:+.1f}%, predicted {(1 - pred_p4 / pred_p0) * 100:+.1f}%.", ""]
        out[prec]["pred"] = {"p4": pred_p4, "p0": pred_p0}
        # host re-tokenization and the patch histogram of P4
        rows = [row for r in runs["p4"] for row in r["rows"]]
        calls = sum(x.get("retok_calls", 0) for x in rows)
        if calls:
            us = sum(x["retok_us_per_call"] * x["retok_calls"] for x in rows) / calls
            ph: dict = {}
            for x in rows:
                for k, v in x.get("patches_hist", {}).items():
                    ph[int(k)] = ph.get(int(k), 0) + v
            md += [f"P4 host re-tokenization: {us:.0f} us per flagged lookup round ({calls} calls); patches per crossing round: "
                   + ", ".join(f"{k}: {v}" for k, v in sorted(ph.items())) + ".", ""]
            out[prec]["retok_us"] = us
            out[prec]["patches_hist"] = ph
    (ROOT / "results/p4/speed_report.md").write_text("\n".join(md), encoding="utf-8")
    (ROOT / "results/p4/speed_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
