"""Task B, Stage B0.2/B0.3: record the small LM's greedy continuation at every position of the ground truth.

For each prompt and each index i (the context = prompt + a_0 .. a_i of the recorded bf16 truth, or the OLMo 2 tokens
t_0 .. t_i) the drafter continues the text by M tokens; the continuations are stored and replayed by
tools/cross_sim_report.py for every K and flag mode (a draft depends on the context only, never on K).

Usage: python tools/cross_record_drafts.py --name smollm2-135m [--n 256] [--m 12]
Output: results/crossing/b0/drafts/<name>.pkl
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from byteprof.drafting.flags import Dolma2  # noqa: E402
from byteprof.drafting.lmdraft import LMEngine, merge_healed, record_continuations  # noqa: E402
from byteprof.drafting.truthio import load_truth  # noqa: E402

DRAFTERS = {
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(DRAFTERS))
    ap.add_argument("--n", type=int, default=256, help="new bytes of the Bolmo truth")
    ap.add_argument("--olmo-tokens", type=int, default=72)
    ap.add_argument("--m", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    ap.add_argument("--no-olmo", action="store_true")
    ap.add_argument("--truth-tag", default="bf16", help="which recorded Bolmo truth (bf16 or int4); int4 skips OLMo")
    a = ap.parse_args()
    repo, rev = DRAFTERS[a.name]
    t0 = time.time()
    dol = Dolma2()
    eng = LMEngine(repo, rev)
    cases = load_truth(ROOT / f"results/drafting/truth/{a.truth_tag}.json")
    olmo = json.loads((ROOT / "results/drafting/truth/olmo-bf16.json").read_text(encoding="utf-8"))["prompts"]
    res = {"name": a.name, "repo": repo, "revision": rev, "params": eng.n_params, "m": a.m, "n": a.n, "bolmo": {}, "olmo": {}}
    memo: dict = {}
    ids = list(cases)[: a.limit or None]
    for pid in ids:
        c = cases[pid]
        atoms = c.truth[a.n]
        prompt = c.prompt_bytes
        ctxs = [prompt + bytes(x & 255 for x in atoms[: i + 1]) for i in range(len(atoms))]
        raw = record_continuations(eng, ctxs, False, a.m, memo)
        healed = record_continuations(eng, ctxs, True, a.m, memo)
        res["bolmo"][pid] = {"raw": raw, "healed": merge_healed(raw, healed, ctxs)}
        if not a.no_olmo and a.truth_tag == "bf16" and pid in olmo:
            o = olmo[pid]
            toks = o["tokens"][str(a.olmo_tokens)]
            pb = o["text"].encode("utf-8")
            octx = [pb + b"".join(dol.id_bytes(t) for t in toks[: i + 1]) for i in range(len(toks))]
            oraw = record_continuations(eng, octx, False, a.m, memo)
            ohealed = record_continuations(eng, octx, True, a.m, memo)
            res["olmo"][pid] = {"raw": oraw, "healed": merge_healed(oraw, ohealed, octx), "prompt_bytes": pb,
                                "tokens": toks}
        print(f"{pid}: {len(ctxs)} contexts, memo {len(memo)}, {time.time() - t0:.0f}s", flush=True)
    res["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2**20
    res["seconds"] = time.time() - t0
    out = ROOT / "results/crossing/b0/drafts"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{a.name}{'' if a.truth_tag == 'bf16' else '-' + a.truth_tag}{'_smoke' if a.limit else ''}.pkl", "wb") as f:
        pickle.dump(res, f)
    print(f"DONE {a.name} params {eng.n_params / 1e6:.0f}M peak VRAM {res['peak_vram_mib']:.0f} MiB {res['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
