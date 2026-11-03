"""Task B, Stage B0.4: GPU cost of a small-LM drafter round in a CUDA graph (static KV cache, no host sync inside).

A drafter round = catch-up forward over the c newly accepted tokens (its last row gives the first draft token)
+ M - 1 single-token greedy steps, all in one graph; the next token feeds the next step on the device.
Timing: CUDA events around R back-to-back replays, median of 5 trials (bf16, batch 1, cache length 512,
context ~200 tokens). The graph starts by resetting the cache position from a base buffer, so replays are idempotent.

Usage: python tools/cross_drafter_cost.py --name smollm2-135m
Output: results/crossing/b0/cost_<name>.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import StaticCache  # noqa: E402

from byteprof.drafting.lmdraft import LMEngine  # noqa: E402

DRAFTERS = {
    "smollm2-135m": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
}
MAXLEN = 512
CMAX = 16
CTX = 200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(DRAFTERS))
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--ms", default="1,2,3,4,6,8,10,12")
    ap.add_argument("--cs", default="1,2,3,4,6,8,12")
    a = ap.parse_args()
    eng = LMEngine(*DRAFTERS[a.name])
    model, dev = eng.model, "cuda"
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAXLEN, device=dev, dtype=torch.bfloat16)
    prompt = torch.randint(1000, 5000, (1, CTX), device=dev)
    with torch.inference_mode():
        model(input_ids=prompt, past_key_values=cache, cache_position=torch.arange(CTX, device=dev), use_cache=True)
    # static inputs for up to CMAX tokens at cache positions CTX .. CTX + CMAX - 1
    ids_full = torch.cat([prompt, torch.zeros(1, CMAX, dtype=prompt.dtype, device=dev)], dim=1)
    mi0 = model.prepare_inputs_for_generation(
        ids_full, past_key_values=cache, attention_mask=torch.ones(1, CTX + CMAX, dtype=prompt.dtype, device=dev),
        cache_position=torch.arange(CTX, CTX + CMAX, device=dev), use_cache=True, logits_to_keep=CMAX)
    ids, cp = mi0["input_ids"].clone(), mi0["cache_position"].clone()
    pos = mi0["position_ids"].clone() if mi0.get("position_ids") is not None else None
    mask0 = mi0["attention_mask"]
    mask_keys = list(mask0) if isinstance(mask0, dict) else None  # sliding-window models pass a dict of masks
    mask = (mask0["full_attention"] if mask_keys else mask0).clone()
    is_bool = mask.dtype == torch.bool
    neg = None if is_bool else torch.finfo(mask.dtype).min
    arange = torch.arange(MAXLEN, device=dev)
    arange_c = torch.arange(CMAX, device=dev)
    base = torch.tensor(CTX, device=dev)
    ids.copy_(torch.randint(1000, 5000, ids.shape, device=dev))

    def set_rows(n: int, start) -> None:
        cp[:n].copy_(start + arange_c[:n])
        if pos is not None:
            pos[:, :n].copy_(cp[:n].view(1, -1))
        allowed = arange[None, :] <= cp[:n, None]
        m = mask[:, :, :n, :]
        m.copy_(allowed.view_as(m) if is_bool else torch.where(allowed, 0.0, neg).to(mask.dtype).view_as(m))

    def forward(n: int):
        m = dict(mi0)
        m.update(input_ids=ids[:, :n], cache_position=cp[:n], attention_mask=({k: mask[:, :, :n, :] for k in mask_keys} if mask_keys else mask[:, :, :n, :]),
                 past_key_values=cache,
                 use_cache=True, logits_to_keep=1)
        if pos is not None:
            m["position_ids"] = pos[:, :n]
        out = model(**m, return_dict=True)
        return torch.argmax(out.logits[:, -1, :].to(torch.float32), dim=-1)

    def chain(c: int, M: int) -> None:
        set_rows(c, base)
        tok = forward(c)
        for j in range(1, M):
            ids[:, :1].copy_(tok.view(1, 1))
            set_rows(1, base + c + j - 1)
            tok = forward(1)

    def capture(c: int, M: int):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(2):
                chain(c, M)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            chain(c, M)
        return g

    def time_graph(g, reps: int) -> float:
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        trials = []
        for _ in range(5):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(reps):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            trials.append(e0.elapsed_time(e1) / reps)
        return statistics.median(trials)

    res = {"name": a.name, "repo": eng.repo, "revision": eng.revision, "params": eng.n_params, "ctx_tokens": CTX,
           "max_cache_len": MAXLEN, "reps": a.reps, "ms": {}}
    t0 = time.time()
    for c in [int(x) for x in a.cs.split(",")]:
        for M in [int(x) for x in a.ms.split(",")]:
            g = capture(c, M)
            with torch.inference_mode():
                res["ms"][f"{c},{M}"] = time_graph(g, a.reps)
            del g
        print(c, {M: round(res["ms"][f"{c},{M}"], 3) for M in [int(x) for x in a.ms.split(",")]}, flush=True)
    res["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2**20
    res["seconds"] = time.time() - t0
    out = ROOT / "results/crossing/b0"
    (out / f"cost_{a.name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("DONE", a.name, f"peak VRAM {res['peak_vram_mib']:.0f} MiB", f"{res['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
