"""Drafter diagnostics, analysis 3: drafter continuations at EVERY byte position of OLMo 2's greedy output.

Task B recorded OLMo's text only at token boundaries (what OLMo's token rule needs). Evaluating Bolmo's byte rule
on the same text needs a continuation after every byte, because a byte round can start anywhere. This records them
(the only GPU use of the diagnostics task, besides tools/diag_ce.py).

The byte text is the prompt + t0 (OLMo's first generated token, "already emitted" in the token rule) followed by
t1 .. t71; context i = prompt + t0 + the first i bytes of t1 .. t71.

Usage: python tools/diag_record_olmo_bytes.py --name smollm2-360m
Output: results/diagnostics/drafts/<name>-olmo-bytes.pkl
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

from byteprof.drafting.flags import Dolma2  # noqa: E402
from byteprof.drafting.lmdraft import LMEngine, merge_healed, record_continuations  # noqa: E402
from cross_record_drafts import DRAFTERS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(DRAFTERS))
    ap.add_argument("--olmo-tokens", type=int, default=72)
    ap.add_argument("--m", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    repo, rev = DRAFTERS[a.name]
    t0 = time.time()
    dol = Dolma2()
    eng = LMEngine(repo, rev)
    olmo = json.loads((ROOT / "results/drafting/truth/olmo-bf16.json").read_text(encoding="utf-8"))["prompts"]
    res = {"name": a.name, "repo": repo, "revision": rev, "m": a.m, "cases": {}}
    memo: dict = {}
    for pid in list(olmo)[: a.limit or None]:
        o = olmo[pid]
        toks = o["tokens"][str(a.olmo_tokens)]
        prompt2 = o["text"].encode("utf-8") + dol.id_bytes(toks[0])
        out = b""
        ends = []  # byte index (in `out`) of each emitted token's end
        for t in toks[1:]:
            out += dol.id_bytes(t)
            ends.append(len(out))
        ctxs = [prompt2 + out[:i] for i in range(len(out) + 1)]
        raw = record_continuations(eng, ctxs, False, a.m, memo)
        healed = record_continuations(eng, ctxs, True, a.m, memo)
        res["cases"][pid] = {"raw": raw, "healed": merge_healed(raw, healed, ctxs), "prompt2": prompt2, "out": out,
                             "token_ends": ends}
        print(f"{pid}: {len(ctxs)} contexts, memo {len(memo)}, {time.time() - t0:.0f}s", flush=True)
    res["seconds"] = time.time() - t0
    res["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2**20
    out_dir = ROOT / "results/diagnostics/drafts"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{a.name}-olmo-bytes{'_smoke' if a.limit else ''}.pkl", "wb") as f:
        pickle.dump(res, f)
    print(f"DONE {a.name} peak VRAM {res['peak_vram_mib']:.0f} MiB {res['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
