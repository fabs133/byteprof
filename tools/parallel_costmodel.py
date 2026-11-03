"""Stage A2 step 2: the drafting cost model of tools/draft_costmodel.py rerun with the round costs of this task
(`steps` and `fused` verification, K up to 32) and the Stage 0 acceptance rates of the drafting task (extended to
K = 12..32). Same model, same instrumented minimal-loop terms (outside-replays time per round, prefill share, global
pass B and C_p - C_n) as the drafting task; only the A_K + C_K round costs come from results/parallel/bench.

Usage: python tools/parallel_costmodel.py   -> results/parallel/costmodel.{json,md}
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from byteprof.drafting.drafters import CombinedDrafter, ContextDrafter, NullDrafter  # noqa: E402
from byteprof.drafting.ngram import NGramDrafter, NGramModel  # noqa: E402
from byteprof.drafting.sim import simulate  # noqa: E402
from byteprof.drafting.truthio import load_truth  # noqa: E402

S1 = ROOT / "results" / "drafting" / "stage1"
BENCH = ROOT / "results" / "parallel" / "bench"
KS = (0, 1, 2, 4, 6, 8, 12, 16, 24, 32)
KMAX = (2, 4, 6, 8, 12, 16, 24, 32)
BENCH_TAGS = {("bf16", "steps"): "bf16-steps", ("bf16", "fused"): "bf16-fused-L0",
              ("int4", "steps"): "int4-steps", ("int4", "fused"): "int4-fused-L0"}


def latest(d: Path, name: str) -> Path:
    return sorted(d.glob(f"*/{name}"))[-1]


def main() -> int:
    stage0 = json.loads((ROOT / "results/drafting/stage0/stage0.json").read_text(encoding="utf-8"))
    ch, host_us = stage0["chosen"], stage0["host_cost_us"]
    corpus = ROOT / "results/drafting/corpus"
    model = NGramModel.build((corpus / "corpus.bin").read_bytes(), np.fromfile(corpus / "flags.bin", np.uint8),
                             (1, 2, 3, 4, 5, 6, 8, 12), bits=24)
    c = ch["combined"]

    def make():
        return CombinedDrafter(ContextDrafter(min_len=c["min_len"]),
                               NGramDrafter(model, min_count=c["min_count"], max_order=c["max_order"]))

    def host(kmax: int) -> float:  # us per round, measured for K_max 1 / 4 / 8; larger K reuses the K = 8 value
        return host_us[f"combined|{1 if kmax <= 1 else 4 if kmax <= 4 else 8}"]["median_us"]

    out: dict = {}
    md = ["# Stage A2: cost model with the round costs of TASK_parallel_verify.md", ""]
    for prec in ("bf16", "int4"):
        tim = json.loads(latest(S1 / f"timing-{prec}", "breakdown.json").read_text(encoding="utf-8"))["rows"]
        ins = [r["instr"] for r in tim]
        nbytes = sum(r["new_bytes"] for r in tim)
        steps_n = sum(i["decode_steps"] for i in ins)
        prefill_pb = sum(i["prefill_ms"] for i in ins) / nbytes
        o_round = sum(i["outside_replays_ms"] for i in ins) / steps_n
        truth = load_truth(ROOT / f"results/drafting/truth/{prec}.json")
        sets = {"default": [x for x in truth.values() if x.set == "default"],
                "ext": [x for x in truth.values() if x.set == "drafting_ext"]}
        rates: dict = {}
        for kmax in KMAX:
            for sname, cases in sets.items():
                tot = {"E": 0, "R": 0, "G": 0, "by_k": {}}
                for cs in cases:
                    r = simulate(cs.truth[256], make(), kmax, ctx_atoms=cs.ctx_atoms, max_new=256,
                                 ks=[k for k in KS if k <= kmax])
                    tot["E"] += r.emitted
                    tot["R"] += r.rounds
                    tot["G"] += r.global_passes
                    for k, n in r.by_k.items():
                        tot["by_k"][k] = tot["by_k"].get(k, 0) + n
                rates[(kmax, sname)] = tot
        md += [f"## {prec}", "", f"Terms outside the local round: prefill {prefill_pb:.4f} ms/byte, outside-replays "
               f"{o_round * 1000:.1f} us per round (drafting task, instrumented minimal loop).", ""]
        out[prec] = {}
        for verify in ("steps", "fused"):
            bench = json.loads((BENCH / BENCH_TAGS[(prec, verify)] / "bench.json").read_text(encoding="utf-8"))
            pair = {int(k): v["ms"] for k, v in bench["rounds"].items()}
            B = bench["pieces"]["B"]["ms"]
            dp = bench["pieces"]["Cp|0"]["ms"] - bench["pieces"]["Cn|0"]["ms"]
            md += [f"### verify = {verify}", "", "| k_max | B/round (default) | predicted ms/byte (default) | gain vs K = 0 | "
                   "B/round (ext) | predicted ms/byte (ext) | gain vs K = 0 (ext) |", "|---:|---:|---:|---:|---:|---:|---:|"]
            res: dict = {}
            for sname in ("default", "ext"):
                base = rates[(KMAX[0], sname)]

                def pred(tot, kmax, null=False):
                    e = tot["E"]
                    local = sum(n * pair[k] for k, n in tot["by_k"].items()) / e
                    glob = tot["G"] * (B + dp) / e
                    hst = tot["R"] * (o_round + (0 if null else host(kmax) / 1000)) / e
                    return prefill_pb + local + glob + hst, e / tot["R"]

                null_tot = {"E": 0, "R": 0, "G": 0, "by_k": {}}
                for cs in sets[sname]:
                    r = simulate(cs.truth[256], NullDrafter(), 0, ctx_atoms=cs.ctx_atoms, max_new=256, ks=[0])
                    null_tot["E"] += r.emitted
                    null_tot["R"] += r.rounds
                    null_tot["G"] += r.global_passes
                    null_tot["by_k"][0] = null_tot["by_k"].get(0, 0) + r.rounds
                p0, _ = pred(null_tot, 0, null=True)
                for kmax in KMAX:
                    p, bpr = pred(rates[(kmax, sname)], kmax)
                    res[f"{sname}|{kmax}"] = {"ms_per_byte": p, "bytes_per_round": bpr, "gain": 1 - p / p0, "k0_ms_per_byte": p0}
            for kmax in KMAX:
                d, e = res[f"default|{kmax}"], res[f"ext|{kmax}"]
                md.append(f"| {kmax} | {d['bytes_per_round']:.2f} | {d['ms_per_byte']:.3f} | {d['gain'] * 100:+.1f}% | "
                          f"{e['bytes_per_round']:.2f} | {e['ms_per_byte']:.3f} | {e['gain'] * 100:+.1f}% |")
            best_ext = max(KMAX, key=lambda k: res[f"ext|{k}"]["gain"])
            best_def = max(KMAX, key=lambda k: res[f"default|{k}"]["gain"])
            res["best_k_ext"], res["best_k_default"] = best_ext, best_def
            md += ["", f"Best k_max by the extended set: {best_ext}; by the default prompts: {best_def}. "
                   f"Predicted K = 0 loop on the default prompts: {res['default|2']['k0_ms_per_byte']:.3f} ms/byte.", ""]
            out[prec][verify] = res
    (ROOT / "results" / "parallel" / "costmodel.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (ROOT / "results" / "parallel" / "costmodel.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
