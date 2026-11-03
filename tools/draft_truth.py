"""Stage 0, step 1: record the greedy ground truth of the existing minimal loop (atoms = byte + boundary flag).

Usage: python tools/draft_truth.py --config configs/fair/bolmo-bf16.toml --tag bf16 --lengths 256,1024 [--eager]

For every prompt of prompts/default.jsonl and prompts/drafting_ext.jsonl it records
  ids        the prompt token ids (specials + byte ids, byte b = id - 4)
  mask       the prefill boundary mask for the prompt bytes except the last (aligned with ids[:-1])
  raw[n]     a_0 .. a_n as raw model ids (plain byte id = byte + 4, boundary-flagged = plain + 260)
at each length n (the 256-byte run is checked to be a prefix of the longest run).
With --eager the HF eager generate() output (bytes) is also recorded, as the H26 reference.
Output: results/drafting/truth/<tag>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from byteprof.config import load_config  # noqa: E402
from byteprof.runner import (_MaxAtoms, generate_checked, load_model, load_prompts, set_seed)  # noqa: E402

OFFSET, BOUNDARY = 4, 260  # plain id = byte + 4; flagged id = plain + 260


class Stop(Exception):
    pass


def prefill_mask(model, tok, ids):
    bm = model.model
    tk = bm.tokenizer
    pad = tk.pad_token_id
    expanded = torch.tensor(tk.expand_byte_ids(ids[0].tolist()), device=ids.device, dtype=torch.long)[None]
    ssi = (ids == pad).sum(-1)
    bm.local_encoder.free_inference_cache()
    bm.local_decoder.free_inference_cache()
    with torch.inference_mode():
        m = bm.prefill_boundary_prediction_forward(ids, expanded_input_ids=expanded, sequence_start_indices=ssi)
    return [int(x) for x in m[0, :-1].tolist()]


def record(model, tok, handle, prompt: str, n: int, patch_limit: int):
    """Runs the minimal loop and returns (raws a_0..a_n, ids, mask, finished_early_reason)."""
    from byteprof.optim import bolmo_loop
    from transformers import StoppingCriteriaList

    cfg_dev = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(cfg_dev)
    mask = prefill_mask(model, tok, ids)
    runner, glob = handle.runner, handle.glob
    names: list[str] = []
    raws: list[int] = []
    orig_run = runner.run

    def run(name):
        names.append(name)
        orig_run(name)

    def dbg(i, info):
        raws.append(info["raw"])
        if int(glob.pos.item()) >= patch_limit:
            raise Stop("static global cache nearly full")

    gen = bolmo_loop.make_generate(model, tok, runner, handle.loop, glob, handle.pieces, handle.automaton, debug=dbg)
    runner.run = run
    why = None
    try:
        with torch.inference_mode():
            out = gen(ids, max_new_tokens=n, stopping_criteria=StoppingCriteriaList([_MaxAtoms(n)]))
    except Stop as e:
        out, why = None, str(e)
    finally:
        runner.run = orig_run
    forced = int(ids[0, -1])
    first_flag = int("glob" in names[:3])  # step 0 runs enc_m, [glob], dec_*
    a0 = forced + (BOUNDARY if first_flag else 0)
    return [a0] + raws, ids[0].tolist(), mask, why, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lengths", default="256,1024")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--prompts", default="prompts/default.jsonl,prompts/drafting_ext.jsonl")
    ap.add_argument("--patch-limit", type=int, default=500)
    a = ap.parse_args()
    lengths = sorted(int(x) for x in a.lengths.split(","))
    cfg = load_config(ROOT / a.config)
    prompts = []
    for f in a.prompts.split(","):
        for p in load_prompts(ROOT / f):
            prompts.append({**p, "set": Path(f).stem})
    model, tok = load_model(cfg)
    sync = torch.cuda.synchronize
    out: dict = {"config": a.config, "tag": a.tag, "dtype": cfg.dtype, "lengths": lengths, "prompts": {}}
    eager: dict[int, dict[str, bytes]] = {}
    if a.eager:
        set_seed(cfg.seed)
        for n in lengths:
            c = load_config(ROOT / a.config, {"max_new_bytes": n})
            t0 = time.time()
            eager[n] = {p["id"]: generate_checked(model, tok, p["text"], c, sync, want_text=True).text for p in prompts}
            print(f"eager n={n} done {time.time() - t0:.0f}s", flush=True)
    from byteprof import optim

    handle = optim.apply(model, tok, cfg)
    for k, p in enumerate(prompts):
        rec = {"set": p["set"], "type": p.get("type") or p["id"].split("_")[0], "text": p["text"], "raw": {}}
        for n in lengths:
            raws, ids, mask, why, gen_out = record(model, tok, handle, p["text"], n, a.patch_limit)
            rec["ids"], rec["mask"] = ids, mask
            rec["raw"][str(n)] = raws
            if why:
                rec.setdefault("stopped_early", {})[str(n)] = why
            if a.eager:
                text = bytes((r % BOUNDARY if r >= BOUNDARY else r) - OFFSET for r in raws[1:] if (r % BOUNDARY if r >= BOUNDARY else r) >= OFFSET)
                ref = eager[n][p["id"]]
                m = min(len(text), len(ref))
                div = next((i for i in range(m) if text[i] != ref[i]), None if len(text) == len(ref) else m)
                rec.setdefault("eager_divergence", {})[str(n)] = div
        out["prompts"][p["id"]] = rec
        r = rec["raw"][str(lengths[-1])]
        flags = sum(1 for x in r[1:] if x >= BOUNDARY)
        print(f"{k + 1}/{len(prompts)} {p['id']:<12} n={len(r) - 1} patches={flags} "
              f"{(len(r) - 1) / max(flags, 1):.2f} B/patch early={rec.get('stopped_early')} "
              f"eager_div={rec.get('eager_divergence')}", flush=True)
    # the short run must be a prefix of the long one
    for pid, rec in out["prompts"].items():
        long = rec["raw"][str(lengths[-1])]
        for n in lengths[:-1]:
            short = rec["raw"][str(n)]
            if short != long[:len(short)]:
                rec.setdefault("prefix_mismatch", []).append(n)
    outdir = ROOT / "results" / "drafting" / "truth"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{a.tag}.json").write_text(json.dumps(out), encoding="utf-8")
    bad = [pid for pid, r in out["prompts"].items() if r.get("prefix_mismatch")]
    print("prefix mismatches:", bad)
    handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
