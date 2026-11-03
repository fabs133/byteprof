"""Drafter diagnostics: re-record drafter continuations with token healing that keeps the leading space.

Task B's `healed` mode holds back the trailing word fragment but leaves the preceding space in the context, so the
drafter's context ends in a bare space and the continuation rarely starts with the fragment (the merge then falls
back to the raw continuation). `healed2` holds back " fragment" (diagnostics.context_back_space). It is recorded for
the Bolmo text (every byte) and the OLMo text (every byte; its token-boundary contexts are a subset).

Usage: python tools/diag_record_healed2.py --name smollm2-360m
Output: results/diagnostics/drafts/<name>-healed2.pkl
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

from byteprof.drafting.diagnostics import context_back_space  # noqa: E402
from byteprof.drafting.flags import Dolma2  # noqa: E402
from byteprof.drafting.lmdraft import LMEngine, merge_healed, record_continuations  # noqa: E402
from byteprof.drafting.truthio import load_truth  # noqa: E402
from cross_record_drafts import DRAFTERS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(DRAFTERS))
    ap.add_argument("--m", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    repo, rev = DRAFTERS[a.name]
    t0 = time.time()
    dol = Dolma2()
    eng = LMEngine(repo, rev)
    cases = load_truth(ROOT / "results/drafting/truth/bf16.json")
    olmo = json.loads((ROOT / "results/drafting/truth/olmo-bf16.json").read_text(encoding="utf-8"))["prompts"]
    ob_path = ROOT / f"results/diagnostics/drafts/{a.name}-olmo-bytes.pkl"
    ob = pickle.load(open(ob_path, "rb"))["cases"]
    res = {"name": a.name, "bolmo": {}, "olmo": {}}
    memo: dict = {}
    for pid in [p for p, c in cases.items() if c.set != "default"][: a.limit or None]:
        c = cases[pid]
        atoms = c.truth[256]
        ctxs = [c.prompt_bytes + bytes(x & 255 for x in atoms[: i + 1]) for i in range(len(atoms))]
        raw = ob_raw = None
        raw = record_continuations(eng, ctxs, False, a.m, memo)  # memoized: only the held-back variants cost GPU time
        h2 = record_continuations(eng, ctxs, True, a.m, memo, back_fn=context_back_space)
        res["bolmo"][pid] = merge_healed(raw, h2, ctxs)
        o = ob[pid]
        octx = [o["prompt2"] + o["out"][:i] for i in range(len(o["out"]) + 1)]
        oraw = record_continuations(eng, octx, False, a.m, memo)
        oh2 = record_continuations(eng, octx, True, a.m, memo, back_fn=context_back_space)
        res["olmo"][pid] = merge_healed(oraw, oh2, octx)
        print(f"{pid}: memo {len(memo)}, {time.time() - t0:.0f}s", flush=True)
    res["seconds"] = time.time() - t0
    res["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2**20
    out = ROOT / "results/diagnostics/drafts"
    with open(out / f"{a.name}-healed2{'_smoke' if a.limit else ''}.pkl", "wb") as f:
        pickle.dump(res, f)
    print(f"DONE {a.name} {res['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
