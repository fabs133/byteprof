"""Task B, Stage B0.3: replay recorded LM continuations (tools/cross_record_drafts.py) in the round simulator.

Usage: python tools/cross_sim_report.py --names smollm2-135m,smollm2-360m,qwen2.5-0.5b [--baseline]
Output: results/crossing/b0/sim_<name>.json (and sim.md via tools/cross_sim_md.py)

Cells: drafter x ctx mode (raw/healed) x flag mode (cut/full) x K, multi-patch rule (at most 8 patches per round,
a0's included) and the within-patch rule for comparison; OLMo 2 token-level acceptance with the same continuations.
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from byteprof.drafting import flag_of  # noqa: E402
from byteprof.drafting.flags import Dolma2  # noqa: E402
from byteprof.drafting.lmdraft import Cont, PrecomputedLMDrafter, PrecomputedTokenDrafter, build_draft  # noqa: E402
from byteprof.drafting.sim import simulate  # noqa: E402
from byteprof.drafting.tokens import simulate_tokens  # noqa: E402
from byteprof.drafting.truthio import load_truth  # noqa: E402

KS = (4, 8, 12, 16, 24, 32)
K_TOK = {4: 1, 8: 2, 12: 3, 16: 4, 24: 6, 32: 8}
MAX_PATCHES = 8
OUT = ROOT / "results/crossing/b0"


class Crossing(PrecomputedLMDrafter):
    """Counts rounds that accepted a flagged draft atom (the round crossed at least one patch boundary)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rounds = self.crossing = 0

    def push(self, atoms):
        if self.i > 0 or len(atoms):
            self.rounds += 1
            self.crossing += any(flag_of(x) for x in atoms[:-1])
        super().push(atoms)


class Retokenized:
    """Baseline: any byte drafter (the existing combined drafter), flags from the dolma2 retokenization."""

    name = "combined+dolma2-flags"

    def __init__(self, inner, dol, flag_mode):
        self.inner, self.dol, self.flag_mode = inner, dol, flag_mode
        self.ctx = bytearray()
        self.rounds = self.crossing = 0

    def reset(self, ctx_atoms, truth=None):
        self.inner.reset(ctx_atoms, truth)
        self.ctx = bytearray(a & 255 for a in ctx_atoms)

    def propose(self, k):
        d = self.inner.propose(k)
        if not d:
            return []
        cont = bytes(a & 255 for a in d)
        return build_draft(bytes(self.ctx), Cont(0, cont, [len(cont)]), k, self.dol, self.flag_mode)[0]

    def push(self, atoms):
        self.rounds += 1
        self.crossing += any(flag_of(x) for x in atoms[:-1])
        self.inner.push(atoms)
        self.ctx.extend(a & 255 for a in atoms)


def pool(rows):
    e = sum(r["emitted"] for r in rows)
    n = sum(r["rounds"] for r in rows)
    g = sum(r["global"] for r in rows)
    cr = sum(r["crossing"] for r in rows)
    return {"emitted": e, "rounds": n, "global": g, "bytes_per_round": e / max(n, 1),
            "bytes_per_global": e / g if g else None, "crossing_share": cr / max(n, 1)}


def run_bolmo(make, cases, k, multi, n=256):
    rows = []
    by_kp: dict = defaultdict(int)
    m_hist: dict = defaultdict(int)
    for pid, c in cases.items():
        truth = c.truth[n]
        d = make(pid)
        res = simulate(truth, d, k, ctx_atoms=c.ctx_atoms, max_new=len(truth) - 1, multi_patch=multi,
                       max_patches=MAX_PATCHES if multi else None)
        rows.append({"id": pid, "type": c.type, "set": c.set, "emitted": res.emitted, "rounds": res.rounds,
                     "global": res.global_passes, "crossing": getattr(d, "crossing", 0)})
        for key, v in res.by_kp.items():
            by_kp[f"{key[0]},{key[1]}"] += v
        for m, v in getattr(d, "m_hist", {}).items():
            m_hist[m] += v
    return rows, dict(by_kp), dict(m_hist)


def summarize(rows):
    out = {"ext": pool([r for r in rows if r["set"] != "default"]), "default": pool([r for r in rows if r["set"] == "default"])}
    types = defaultdict(list)
    for r in rows:
        if r["set"] != "default":
            types[r["type"]].append(r)
    out["by_type"] = {t: pool(v)["bytes_per_round"] for t, v in types.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    dol = Dolma2()
    cases = load_truth(ROOT / "results/drafting/truth/bf16.json")
    OUT.mkdir(parents=True, exist_ok=True)

    if a.baseline:
        from byteprof.drafting.drafters import CombinedDrafter, ContextDrafter
        from byteprof.drafting.ngram import NGramDrafter, NGramModel

        data = (ROOT / "results/drafting/corpus/corpus.bin").read_bytes()
        flags = np.fromfile(ROOT / "results/drafting/corpus/flags.bin", np.uint8)
        model = NGramModel.build(data, flags, (1, 2, 3, 4, 5, 6, 8, 12), bits=24)
        res = {"name": "combined", "cells": {}}
        for fm in ("cut", "full"):
            for k in KS:
                for multi in (True, False):
                    make = lambda pid, fm=fm: Retokenized(CombinedDrafter(ContextDrafter(min_len=3), NGramDrafter(model, min_count=1, max_order=6)), dol, fm)
                    rows, by_kp, _ = run_bolmo(make, cases, k, multi, a.n)
                    res["cells"][f"{fm}|{k}|{'multi' if multi else 'single'}"] = {**summarize(rows), "by_kp": by_kp}
                    print("baseline", fm, k, multi, round(res["cells"][f"{fm}|{k}|{'multi' if multi else 'single'}"]["ext"]["bytes_per_round"], 3), flush=True)
        (OUT / "sim_combined.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    for name in [x for x in a.names.split(",") if x]:
        with open(OUT / "drafts" / f"{name}.pkl", "rb") as f:
            rec = pickle.load(f)
        res = {"name": name, "repo": rec["repo"], "revision": rec["revision"], "params": rec["params"], "cells": {},
               "olmo": {}}
        for cm in ("raw", "healed"):
            for fm in ("cut", "full"):
                for k in KS:
                    for multi in (True, False):
                        make = lambda pid, cm=cm, fm=fm: Crossing(rec["bolmo"][pid][cm], dol, fm)
                        rows, by_kp, mh = run_bolmo(make, cases, k, multi, a.n)
                        res["cells"][f"{cm}|{fm}|{k}|{'multi' if multi else 'single'}"] = {**summarize(rows), "by_kp": by_kp, "m_hist": mh}
                    c = res["cells"][f"{cm}|{fm}|{k}|multi"]["ext"]
                    print(name, cm, fm, k, "multi bytes/round", round(c["bytes_per_round"], 3), "cross", round(c["crossing_share"], 3), flush=True)
        # OLMo 2: token-level acceptance with the same continuations
        for cm in ("raw", "healed"):
            for k in KS:
                e = r = tb = 0
                eb = 0
                e_ext = r_ext = eb_ext = 0
                mh: dict = defaultdict(int)
                for pid, o in rec["olmo"].items():
                    toks = o["tokens"]
                    d = PrecomputedTokenDrafter(o[cm], dol)
                    d.set_text(o["prompt_bytes"], dol.id_bytes(toks[0]))
                    s = simulate_tokens(toks, d, K_TOK[k], ctx=[], max_new=len(toks) - 1)
                    nb = sum(len(dol.id_bytes(t)) for t in toks[1:1 + s.emitted])
                    is_ext = cases[pid].set != "default"
                    if is_ext:
                        e_ext += s.emitted
                        r_ext += s.rounds
                        eb_ext += nb
                    for m, v in d.m_hist.items():
                        mh[m] += v
                res["olmo"][f"{cm}|{k}"] = {"tokens_per_round": e_ext / max(r_ext, 1), "bytes_per_round": eb_ext / max(r_ext, 1),
                                            "rounds": r_ext, "emitted_tokens": e_ext, "k_tok": K_TOK[k], "m_hist": dict(mh)}
                print(name, "olmo", cm, k, round(e_ext / max(r_ext, 1), 3), "tok/round", flush=True)
        (OUT / f"sim_{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
