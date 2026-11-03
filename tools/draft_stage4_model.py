"""Stage 4 gate: predicted ms/byte of multi-patch drafting vs within-patch drafting, with B(P) for P = 1..4 patches in one pass.
Same model as draft_costmodel.py; a round costs pair(K) + B(P) with P = patches the round runs through the global model
(multi-patch: a0's plus every flagged drafted atom, speculatively; at most 4). Local-pass extras of the multi-patch decoder
(per-position patch vector select) are not modelled. Output: results/drafting/stage4/stage4_model.{json,md}"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from byteprof.drafting.drafters import CombinedDrafter, ContextDrafter  # noqa: E402
from byteprof.drafting.ngram import NGramDrafter, NGramModel  # noqa: E402
from byteprof.drafting.sim import simulate  # noqa: E402
from byteprof.drafting.truthio import load_truth  # noqa: E402

cm = json.loads((ROOT / "results/drafting/stage1/costmodel.json").read_text(encoding="utf-8"))
ch = cm["chosen"]["combined"]
corpus = ROOT / "results/drafting/corpus"
model = NGramModel.build((corpus / "corpus.bin").read_bytes(), np.fromfile(corpus / "flags.bin", np.uint8), (1, 2, 3, 4, 5, 6, 8, 12), bits=24)
mk = lambda: CombinedDrafter(ContextDrafter(min_len=ch["min_len"]), NGramDrafter(model, min_count=ch["min_count"], max_order=ch["max_order"]))
out, md = {}, ["# Stage 4 gate: multi-patch vs within-patch drafting (predicted)", ""]
for prec in ("bf16", "int4"):
    v = cm[prec]["validation"]
    pair = {int(k): x for k, x in v["microbench"]["pair"].items()}
    gm = json.loads((ROOT / f"results/drafting/stage4/global_multi-{prec}.json").read_text(encoding="utf-8"))
    B = {0: 0.0, **{p: gm[f"P={p}"]["ms"] for p in (1, 2, 3, 4)}}
    dp = v["microbench"]["Cp_minus_Cn"]
    truth = load_truth(ROOT / f"results/drafting/truth/{prec}.json")
    for sname in ("default", "ext"):
        cases = [c for c in truth.values() if c.set == ("default" if sname == "default" else "drafting_ext")]
        def pred(multi, km):
            tot = {"E": 0, "R": 0, "kp": {}}
            for c in cases:
                r = simulate(c.truth[256], mk(), km, ctx_atoms=c.ctx_atoms, max_new=256, multi_patch=multi,
                             ks=[k for k in (0, 1, 2, 4, 6, 8) if k <= km], max_patches=4 if multi else None)
                tot["E"] += r.emitted; tot["R"] += r.rounds
                for key, n in r.by_kp.items(): tot["kp"][key] = tot["kp"].get(key, 0) + n
            loc = sum(n * (pair[k] + B[p] + (dp if p else 0)) for (k, p), n in tot["kp"].items())
            ms = v["prefill_per_byte"] + (loc + tot["R"] * (v["outside_per_round"] + 0.0095)) / tot["E"]
            return ms, tot["E"] / tot["R"]
        base = pred(False, 0)[0]
        md += [f"## {prec}, {sname} prompts (256 new bytes); predicted K = 0 loop {base:.3f} ms/byte", "",
               "| K | within-patch ms/byte | gain | multi-patch ms/byte | B/round | gain | multi gain / within gain | multi ms/byte vs within |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for km in (2, 4, 6, 8):
            s, bs = pred(False, km); m, bm = pred(True, km)
            out[f"{prec}|{sname}|{km}"] = {"within": s, "multi": m, "base": base, "bpr_within": bs, "bpr_multi": bm}
            md.append(f"| {km} | {s:.3f} | {(1 - s / base) * 100:+.1f}% | {m:.3f} | {bm:.2f} | {(1 - m / base) * 100:+.1f}% | {(1 - m / base) / (1 - s / base):.2f}x | {(1 - m / s) * 100:+.1f}% |")
        md.append("")
(ROOT / "results/drafting/stage4").mkdir(exist_ok=True, parents=True)
(ROOT / "results/drafting/stage4/stage4_model.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
(ROOT / "results/drafting/stage4/stage4_model.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
