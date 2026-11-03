"""Patch-aligned task, step 4: simulate the policies on the recorded truth and the recorded drafter continuations.

Usage: python tools/aligned_sim.py --prec bf16 --names lookup            (P0, P1, P4 and the lookup oracles)
       python tools/aligned_sim.py --prec bf16 --names smollm2-135m      (L1, P2, P2b, P3 and the LM oracles)
Output: results/aligned/tallies_<prec>_<name>.json  (one entry per set|ctx|flagmode|K|policy[*]; `*` = oracle)

CPU only. Contexts: raw / healed (Task B recordings, both precisions) and healed2 (diagnostics recording; extended set
and bf16 only, no default prompts and no int4 recording exists).
"""

import argparse
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from byteprof.drafting.aligned import RecordedSources, simulate_policy  # noqa: E402
from byteprof.drafting.drafters import CombinedDrafter, ContextDrafter  # noqa: E402
from byteprof.drafting.flags import Dolma2  # noqa: E402
from byteprof.drafting.ngram import NGramDrafter, NGramModel  # noqa: E402
from byteprof.drafting.truthio import load_truth  # noqa: E402

KS_ALL = (1, 2, 4, 6, 8, 12, 16, 24, 32)
KS_EVAL = (4, 8, 12, 16, 24, 32)  # the task's 8 / 16 / 32 plus the Task B grid
OUT = ROOT / "results/aligned"
LOOKUP_POLICIES = [("P0", False), ("P1", False), ("P4", False), ("P4", True)]
LM_POLICIES = [("L1", False), ("P2", False), ("P2b", False), ("P3", False), ("P2", True), ("P3", True)]


def dump(r) -> dict:
    return {"emitted": r.emitted, "rounds": r.rounds, "global": r.global_passes, "lm_calls": r.lm_calls,
            "lookup_raw_calls": r.lookup_raw_calls, "lookup_flagged_calls": r.lookup_flagged_calls,
            "by_kp": {f"{a},{b},{int(c)},{int(d)}": v for (a, b, c, d), v in r.by_kp.items()},
            "m_hist": {str(k): v for k, v in r.lm_m_hist.items()}, "kinds": r.kinds,
            "start_rounds": {str(k): v for k, v in r.start_rounds.items()}, "start_bytes": {str(k): v for k, v in r.start_bytes.items()},
            "trans": {f"{a},{b}": v for (a, b), v in r.trans.items()}, "accepted": r.accepted, "drafted": r.drafted}


def add(acc: dict, d: dict) -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            sub = acc.setdefault(k, {})
            for kk, vv in v.items():
                sub[kk] = sub.get(kk, 0) + vv
        else:
            acc[k] = acc.get(k, 0) + v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prec", default="bf16", choices=("bf16", "int4"))
    ap.add_argument("--names", default="lookup")
    ap.add_argument("--flagmodes", default="cut,full")
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
    for name in [x for x in a.names.split(",") if x]:
        out: dict = {"prec": a.prec, "cells": {}}
        t0 = time.time()
        if name == "lookup":
            ctxs = {"-": None}
            pols = LOOKUP_POLICIES
        else:
            if a.prec == "bf16":
                rec = pickle.load(open(ROOT / f"results/crossing/b0/drafts/{name}.pkl", "rb"))["bolmo"]
                h2 = pickle.load(open(ROOT / f"results/diagnostics/drafts/{name}-healed2.pkl", "rb"))["bolmo"]
            else:
                rec = pickle.load(open(ROOT / f"results/crossing/b0/drafts/{name}-int4.pkl", "rb"))["bolmo"]
                h2 = None
            ctxs = {"raw": lambda pid: rec[pid]["raw"], "healed": lambda pid: rec[pid]["healed"]}
            if h2 is not None:
                ctxs["healed2"] = lambda pid: h2[pid]
            pols = LM_POLICIES
        lm_times: list[float] = []
        for cm, getter in ctxs.items():
            for fm in ([x for x in a.flagmodes.split(",") if x] if name != "lookup" else ["cut"]):
                for k in KS_EVAL:
                    for pol, orc in pols:
                        for sname, cs in sets.items():
                            tot: dict = {}
                            for c in cs:
                                truth = c.truth[a.n]
                                if cm == "healed2" and c.id not in h2:
                                    continue
                                items = getter(c.id) if getter else None
                                src = RecordedSources(lookup(), dol, c.ctx_atoms, truth, items, fm)
                                r = simulate_policy(truth, pol, src, k, max_new=len(truth) - 1,
                                                    ks=[x for x in KS_ALL if x <= k], oracle=orc)
                                add(tot, dump(r))
                                lm_times += src.lm_times
                            if tot:
                                out["cells"][f"{sname}|{cm}|{fm}|{k}|{pol}{'*' if orc else ''}"] = tot
                print(name, a.prec, cm, "done", round(time.time() - t0), "s", flush=True)
        out["lm_host_us"] = statistics.median(lm_times) * 1e6 if lm_times else None
        (OUT / f"tallies_{a.prec}_{name}.json").write_text(json.dumps(out), encoding="utf-8")
        print("wrote", name, a.prec, round(time.time() - t0), "s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
