"""OLMo 2 token drafting: acceptance by drafter x K (simulation on the recorded truth), parameters tuned on the
extended prompt set only. Output: results/drafting/stage3/olmo_sim.{json,md}

Usage: python tools/olmo_sim_report.py --truth results/drafting/truth/olmo-bf16.json
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from byteprof.drafting.drafters import CombinedDrafter, OracleDrafter  # noqa: E402
from byteprof.drafting.tokens import (TokenContextDrafter, TokenNGramDrafter, TokenNGramModel,  # noqa: E402
                                      simulate_tokens)

KS = (0, 1, 2, 4, 6, 8)


def pooled(cases, make, k, n):
    e = r = 0
    by_k: dict[int, int] = {}
    for c in cases:
        toks = c["tokens"][str(n)]
        s = simulate_tokens(toks, make(), k, ctx=c["ids"], max_new=len(toks) - 1, ks=[x for x in KS if x <= k])
        e += s.emitted
        r += s.rounds
        for kk, v in s.by_k.items():
            by_k[kk] = by_k.get(kk, 0) + v
    return {"tokens_per_round": e / r, "emitted": e, "rounds": r, "by_k": by_k}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="results/drafting/truth/olmo-bf16.json")
    a = ap.parse_args()
    truth = json.loads((ROOT / a.truth).read_text(encoding="utf-8"))["prompts"]
    ext = [dict(c, id=k) for k, c in truth.items() if c["set"] == "drafting_ext"]
    dflt = [dict(c, id=k) for k, c in truth.items() if c["set"] == "default"]
    tokens = np.load(ROOT / "results/drafting/corpus/olmo_tokens.npy")
    model = TokenNGramModel.build(tokens, (1, 2, 3, 4), bits=22)
    n0 = str(min(int(x) for x in ext[0]["tokens"]))
    n_short = int(n0)

    def ctx(m):
        return lambda: TokenContextDrafter(min_len=m)

    def ng(mc, mo):
        return lambda: TokenNGramDrafter(model, min_count=mc, max_order=mo)

    def comb(m, mc, mo):
        return lambda: CombinedDrafter(TokenContextDrafter(min_len=m), TokenNGramDrafter(model, min_count=mc, max_order=mo))

    tune = {"context": {str(m): pooled(ext, ctx(m), 4, n_short)["tokens_per_round"] for m in (1, 2, 3, 4)},
            "ngram": {f"{mc},{mo}": pooled(ext, ng(mc, mo), 4, n_short)["tokens_per_round"]
                      for mc, mo in itertools.product((1, 2, 3, 5), (2, 3, 4))},
            "combined": {f"{m},{mc},{mo}": pooled(ext, comb(m, mc, mo), 4, n_short)["tokens_per_round"]
                         for m, mc, mo in itertools.product((1, 2, 3, 4), (1, 2, 3, 5), (2, 3, 4))}}
    bc = int(max(tune["context"], key=tune["context"].get))
    bn = tuple(int(x) for x in max(tune["ngram"], key=tune["ngram"].get).split(","))
    bb = tuple(int(x) for x in max(tune["combined"], key=tune["combined"].get).split(","))
    makers = {"context": ctx(bc), "ngram": ng(*bn), "combined": comb(*bb), "oracle (upper bound)": lambda: OracleDrafter()}
    grid = {}
    for sname, cs in (("ext", ext), ("default", dflt)):
        for n in sorted(int(x) for x in ext[0]["tokens"]):
            for d, mk in makers.items():
                for k in KS:
                    grid[f"{sname}|{n}|{d}|{k}"] = pooled(cs, mk, k, n)
    res = {"chosen": {"context_min_len": bc, "ngram_min_count": bn[0], "ngram_max_order": bn[1],
                      "combined": {"min_len": bb[0], "min_count": bb[1], "max_order": bb[2]}}, "tune": tune, "grid": grid}
    out = ROOT / "results" / "drafting" / "stage3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "olmo_sim.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = ["# OLMo 2 token drafting: simulated tokens per round (pooled)", "", f"chosen on the extended set ({n_short} tokens, K = 4): {res['chosen']}", ""]
    for sname in ("ext", "default"):
        for n in sorted(int(x) for x in ext[0]["tokens"]):
            L += [f"## {sname}, {n} tokens", "", "| drafter | " + " | ".join(f"K={k}" for k in KS) + " |", "|---|" + "---:|" * len(KS)]
            for d in makers:
                L.append(f"| {d} | " + " | ".join(f"{grid[f'{sname}|{n}|{d}|{k}']['tokens_per_round']:.3f}" for k in KS) + " |")
            L.append("")
    (out / "olmo_sim.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
