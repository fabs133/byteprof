"""Stage 0 report (TASK_build_p4.md): crossing-round cost by K and P from results/p4/stage0/*.json -> results/p4/stage0/report.md"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results/p4/stage0"


def main() -> int:
    md = ["# Stage 0: one crossing round (encd{K} + gx{P} + decx{K}), GPU ms per round, L = 0", ""]
    summary = {}
    for prec in ("bf16", "int4"):
        cells: dict = {}
        meta = {}
        for f in sorted(D.glob(f"{prec}-k*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            meta = r
            for key, c in r["cells"].items():
                k, p = (int(x) for x in key.split("|"))
                cells[(k, p)] = c
        ks = sorted({k for k, _ in cells})
        md += [f"## {prec}", "",
               f"{meta['reps']} replays per trial, median of {meta['trials']} trials, trials of the compared graphs interleaved. "
               "Cell: X = measured round / extra over `pair(K) + B(P) + dp` measured in the same session / extra over the stored cost-model inputs "
               "(`results/parallel/bench/*-fused-L0`, `results/drafting/stage4/global_multi-*-p8`). P = 1 is the new graphs with one patch (the loop uses the existing path there).", "",
               "| P | " + " | ".join(f"K = {k}" for k in ks) + " |", "|---:|" + "---:|" * len(ks)]
        for p in range(1, 9):
            row = []
            for k in ks:
                c = cells.get((k, p))
                row.append(f"{c['X']:.3f} / {c['extra_now']:+.3f} / {c['extra_stored']:+.3f}" if c else "")
            md.append(f"| {p} | " + " | ".join(row) + " |")
        ex = [c["extra_now"] for (k, p), c in cells.items() if p >= 2]
        exs = [c["extra_stored"] for (k, p), c in cells.items() if p >= 2]
        worst = max(cells.items(), key=lambda kv: kv[1]["extra_now"] if kv[0][1] >= 2 else -9)
        worst_s = max(cells.items(), key=lambda kv: kv[1]["extra_stored"] if kv[0][1] >= 2 else -9)
        y = {k: statistics.mean(cells[(k, p)]["Y"] for p in range(1, 9) if (k, p) in cells) for k in ks}
        md += ["", f"Extra over the model (same session), P >= 2, {len(ex)} cells: mean {statistics.mean(ex):+.3f} ms, max {max(ex):+.3f} ms (K = {worst[0][0]}, P = {worst[0][1]}), min {min(ex):+.3f}; "
               f"against the stored inputs: mean {statistics.mean(exs):+.3f}, max {max(exs):+.3f} (K = {worst_s[0][0]}, P = {worst_s[0][1]}), min {min(exs):+.3f}.",
               "Decomposition at K = {}: `gx{{P}}` minus the plain global pass B(P) (pooling gather + out_projection over P rows), and `decx` minus `decd{{K}}p` (patch select, raw-id accept, rollback):".format(ks[-1]), ""]
        md += ["| P | gx - B(P) | decx - Cp |", "|---:|---:|---:|"]
        for p in range(2, 9):
            c = cells.get((ks[-1], p))
            if c:
                md.append(f"| {p} | {c['Gx'] - c['Bnow']:+.3f} | {c['decx'] - c['Cp']:+.3f} |")
        md.append("")
        summary[prec] = {"mean_extra_now": statistics.mean(ex), "max_extra_now": max(ex), "mean_extra_stored": statistics.mean(exs), "max_extra_stored": max(exs),
                         "worst_now": list(worst[0]), "worst_stored": list(worst_s[0])}
    (D / "report.md").write_text("\n".join(md), encoding="utf-8")
    (D / "report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
