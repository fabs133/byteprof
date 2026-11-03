"""Stage 1, steps 3-5: cost model, K = 0 validation, predictions, and the go/no-go gate.

predicted ms/byte = prefill share
                  + sum over verification sizes K of (rounds_K / bytes) * (A_K + C_K replayed as a round)
                  + (global passes / bytes) * (B + C_p - C_n)
                  + (rounds / bytes) * (outside-replays time per round of the minimal loop + drafter host time)
Rates (rounds per byte, K mix, global passes per byte) come from the Stage 0 simulator run on the recorded
truth of the same precision, on the default prompts at 256 new bytes (the speed-table workload), with the real
loop's rounding to the available graph sizes. Parameters of the drafters were chosen on the extended set.

Usage: python tools/draft_costmodel.py   -> results/drafting/stage1/{costmodel.json,costmodel.md}
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from byteprof.drafting.drafters import CombinedDrafter, ContextDrafter, NullDrafter, OracleDrafter  # noqa: E402
from byteprof.drafting.ngram import NGramDrafter, NGramModel  # noqa: E402
from byteprof.drafting.sim import simulate  # noqa: E402
from byteprof.drafting.truthio import load_truth  # noqa: E402

S1 = ROOT / "results" / "drafting" / "stage1"
KS = (0, 1, 2, 4, 6, 8)
KMAX = (1, 2, 4, 6, 8)
PLAIN_RUNS = {"bf16": [1, 3, 4], "int4": [1, 2]}  # runs used (bf16 run 2 differed by 5.8% from run 1 and was discarded)


def latest(d: Path, name: str) -> Path:
    return sorted(d.glob(f"*/{name}"))[-1]


def plain_ms_per_byte(prec: str, run: int) -> float:
    rows = json.loads(latest(S1 / f"plain-{prec}-{run}", "breakdown.json").read_text(encoding="utf-8"))["rows"]
    return sum(r["ms"] for r in rows) / sum(r["new_bytes"] for r in rows)


def main() -> int:
    stage0 = json.loads((ROOT / "results/drafting/stage0/stage0.json").read_text(encoding="utf-8"))
    ch = stage0["chosen"]
    host_us = stage0["host_cost_us"]
    corpus = ROOT / "results/drafting/corpus"
    model = NGramModel.build((corpus / "corpus.bin").read_bytes(), np.fromfile(corpus / "flags.bin", np.uint8),
                             (1, 2, 3, 4, 5, 6, 8, 12), bits=24)
    makers = {
        "context": lambda: ContextDrafter(min_len=ch["context_min_len"]),
        "ngram": lambda: NGramDrafter(model, min_count=ch["ngram_min_count"], max_order=ch["ngram_max_order"]),
        "combined": lambda: CombinedDrafter(ContextDrafter(min_len=ch["combined"]["min_len"]),
                                            NGramDrafter(model, min_count=ch["combined"]["min_count"],
                                                         max_order=ch["combined"]["max_order"])),
        "oracle (upper bound)": lambda: OracleDrafter(),
    }

    def host_for(d: str, kmax: int) -> float:  # us per round
        if d.startswith("oracle") or d == "none":
            return 0.0 if d == "none" else 1.0
        return host_us[f"{d}|{1 if kmax <= 1 else 4 if kmax <= 4 else 8}"]["median_us"]

    out: dict = {"chosen": ch}
    md: list[str] = ["# Stage 1: cost model, validation and predictions", ""]
    for prec in ("bf16", "int4"):
        bench = json.loads((S1 / prec / "bench.json").read_text(encoding="utf-8"))
        pair = {int(k): v["ms"] for k, v in bench["rounds"].items()}
        B = bench["pieces"]["B"]["ms"]
        dp = bench["pieces"]["Cp|0"]["ms"] - bench["pieces"]["Cn|0"]["ms"]
        tim = json.loads(latest(S1 / f"timing-{prec}", "breakdown.json").read_text(encoding="utf-8"))["rows"]
        ins = [r["instr"] for r in tim]
        nbytes = sum(r["new_bytes"] for r in tim)
        steps = sum(i["decode_steps"] for i in ins)
        prefill_pb = sum(i["prefill_ms"] for i in ins) / nbytes
        o_round = sum(i["outside_replays_ms"] for i in ins) / steps
        gpu = {g: sum(i[f"gpu_ms_{g}"] for i in ins) for g in ("enc", "dec", "glob")}
        instr = {"enc_per_round": gpu["enc"] / steps, "dec_per_round": gpu["dec"] / steps,
                 "glob_per_call": gpu["glob"] / sum(i["glob_calls"] for i in ins),
                 "glob_calls_per_byte": sum(i["glob_calls"] for i in ins) / steps}
        truth = load_truth(ROOT / f"results/drafting/truth/{prec}.json")
        dflt = [c for c in truth.values() if c.set == "default"]
        ext = [c for c in truth.values() if c.set == "drafting_ext"]

        def rates(cases, make, kmax):
            tot = {"E": 0, "R": 0, "G": 0, "by_k": {}}
            for c in cases:
                t = c.truth[256]
                r = simulate(t, make(), kmax, ctx_atoms=c.ctx_atoms, max_new=256, ks=[k for k in KS if k <= kmax])
                tot["E"] += r.emitted
                tot["R"] += r.rounds
                tot["G"] += r.global_passes
                for k, n in r.by_k.items():
                    tot["by_k"][k] = tot["by_k"].get(k, 0) + n
            return tot

        def predict(tot, drafter: str, kmax: int, prefill: float = prefill_pb) -> dict:
            E = tot["E"]
            local = sum(n * pair[k] for k, n in tot["by_k"].items()) / E
            glob = tot["G"] * (B + dp) / E
            host = tot["R"] * (o_round + host_for(drafter, kmax) / 1000) / E
            return {"ms_per_byte": prefill + local + glob + host, "local": local, "global": glob, "host": host,
                    "prefill": prefill, "bytes_per_round": E / tot["R"], "rounds_per_byte": tot["R"] / E,
                    "global_per_byte": tot["G"] / E, "by_k": tot["by_k"]}

        base_tot = rates(dflt, lambda: NullDrafter(), 0)
        pred0 = predict(base_tot, "none", 0)
        measured = [plain_ms_per_byte(prec, i) for i in PLAIN_RUNS[prec]]
        m0 = statistics.mean(measured)
        val = {"predicted_k0": pred0["ms_per_byte"], "measured_runs": measured, "measured_mean": m0,
               "relative_error": pred0["ms_per_byte"] / m0 - 1, "components_k0": pred0,
               "microbench": {"pair": pair, "B": B, "Cp_minus_Cn": dp}, "instrumented": instr,
               "outside_per_round": o_round, "prefill_per_byte": prefill_pb}
        preds: dict = {}
        for d, make in makers.items():
            for km in KMAX:
                for sname, cases in (("default", dflt), ("ext", ext)):
                    tot = rates(cases, make, km)
                    p = predict(tot, d, km)
                    p["gain_vs_k0_pred"] = 1 - p["ms_per_byte"] / pred0["ms_per_byte"] if sname == "default" else None
                    preds[f"{d}|{km}|{sname}"] = p
        out[prec] = {"validation": val, "predictions": preds}

        md += [f"## {prec}", "",
               f"Pieces (GPU ms, microbenchmark): A+C round K = 0: {pair[0]:.4f}; B (global, per patch): {B:.4f}; "
               f"C_p - C_n: {dp * 1000:.1f} us. Instrumented minimal-loop run (25 generations, default prompts): "
               f"enc {instr['enc_per_round']:.4f}, dec {instr['dec_per_round']:.4f} ms per round, global "
               f"{instr['glob_per_call']:.3f} ms per call, {instr['glob_calls_per_byte']:.3f} global calls per byte, "
               f"outside replays {o_round * 1000:.1f} us per round, prefill {prefill_pb:.4f} ms per byte.", "",
               f"**K = 0 validation:** predicted {pred0['ms_per_byte']:.3f} ms/byte (local {pred0['local']:.3f}, global "
               f"{pred0['global']:.3f}, host {pred0['host']:.3f}, prefill {pred0['prefill']:.3f}) against measured "
               f"{m0:.3f} (plain runs {', '.join(f'{x:.3f}' for x in measured)}): **{val['relative_error'] * 100:+.1f}%**.", "",
               "Predicted ms/byte (default prompts, 256 new bytes) and gain over the predicted K = 0 loop; B/round = emitted bytes per round", "",
               "| drafter | k_max | B/round | rounds/byte | K mix | predicted ms/byte | gain |", "|---|---:|---:|---:|---|---:|---:|"]
        for d in makers:
            for km in KMAX:
                p = preds[f"{d}|{km}|default"]
                mix = ", ".join(f"K{k}:{n / sum(p['by_k'].values()):.2f}" for k, n in sorted(p["by_k"].items()))
                md.append(f"| {d} | {km} | {p['bytes_per_round']:.2f} | {p['rounds_per_byte']:.3f} | {mix} | "
                          f"{p['ms_per_byte']:.3f} | {p['gain_vs_k0_pred'] * 100:+.1f}% |")
        md.append("")

    # ---- gate ----
    best = {}
    for prec in ("bf16", "int4"):
        c = {k: v for k, v in out[prec]["predictions"].items() if k.endswith("|default") and not k.startswith("oracle")}
        k, v = max(c.items(), key=lambda kv: kv[1]["gain_vs_k0_pred"])
        best[prec] = (k.rsplit("|", 1)[0], v["gain_vs_k0_pred"], v["ms_per_byte"])
    out["gate"] = {prec: {"best": b[0], "gain": b[1], "ms_per_byte": b[2]} for prec, b in best.items()}
    out["gate"]["passes"] = any(b[1] >= 0.05 for b in best.values())
    md += ["## Gate", "", "Best real drafter (excluding the oracle) by predicted gain: " +
           "; ".join(f"{p}: {b[0]} -> {b[1] * 100:+.1f}% ({b[2]:.3f} ms/byte)" for p, b in best.items()) +
           f". Gate (>= 5% in bf16 or all-int4): {'PASS' if out['gate']['passes'] else 'STOP'}."]
    (S1 / "costmodel.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    (S1 / "costmodel.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
