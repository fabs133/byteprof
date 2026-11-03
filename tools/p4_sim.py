"""Stage 0 step 4 (TASK_build_p4.md): P4 tallies with separate K for mid-patch and crossing rounds (the policy as built:
`draft.k_max` within the patch, `draft.k_cross` across patches), CPU only, on the recorded truth.

Usage: python tools/p4_sim.py --prec bf16     -> results/p4/tallies_<prec>.json
Keys: <set>|<policy>|<k_cross>|<k_within>   (policy P4; k_within = "-" means the same K as the crossing rounds, the
patch-aligned task's P4). Tallies have the layout of tools/aligned_sim.py (read by tools/p4_costmodel.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from aligned_sim import KS_ALL, add, dump  # noqa: E402
from byteprof.drafting.aligned import RecordedSources, simulate_policy  # noqa: E402
from byteprof.drafting.drafters import CombinedDrafter, ContextDrafter  # noqa: E402
from byteprof.drafting.flags import Dolma2  # noqa: E402
from byteprof.drafting.ngram import NGramDrafter, NGramModel  # noqa: E402
from byteprof.drafting.truthio import load_truth  # noqa: E402

OUT = ROOT / "results/p4"
K_CROSS = (8, 12, 16, 24, 32)
K_WITHIN = (None, 4, 6, 8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prec", default="bf16", choices=("bf16", "int4"))
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dol = Dolma2()
    cases = load_truth(ROOT / f"results/drafting/truth/{a.prec}.json")
    corpus = ROOT / "results/drafting/corpus"
    model = NGramModel.build((corpus / "corpus.bin").read_bytes(), np.fromfile(corpus / "flags.bin", np.uint8),
                             (1, 2, 3, 4, 5, 6, 8, 12), bits=24)

    def lookup():
        return CombinedDrafter(ContextDrafter(min_len=3), NGramDrafter(model, min_count=1, max_order=6))

    sets = {"default": [c for c in cases.values() if c.set == "default"], "ext": [c for c in cases.values() if c.set != "default"]}
    out: dict = {"prec": a.prec, "cells": {}}
    t0 = time.time()
    for kc in K_CROSS:
        for kw in K_WITHIN:
            if kw is not None and kw >= kc:
                continue
            for sname, cs in sets.items():
                tot: dict = {}
                for c in cs:
                    truth = c.truth[a.n]
                    src = RecordedSources(lookup(), dol, c.ctx_atoms, truth, None, "cut")
                    r = simulate_policy(truth, "P4", src, kc, max_new=len(truth) - 1, ks=[x for x in KS_ALL if x <= kc], k_within=kw)
                    add(tot, dump(r))
                out["cells"][f"{sname}|P4|{kc}|{kw if kw is not None else '-'}"] = tot
            print(a.prec, "K_cross", kc, "K_within", kw, round(time.time() - t0), "s", flush=True)
    (OUT / f"tallies_{a.prec}.json").write_text(json.dumps(out), encoding="utf-8")
    print("wrote", a.prec, round(time.time() - t0), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
