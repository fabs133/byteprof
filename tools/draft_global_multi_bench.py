"""Stage 4 gate input: time the 16 global layers over P patches in one pass (P = 1..4) on the static KV cache.

Usage: python tools/draft_global_multi_bench.py --config configs/draft/bolmo-bf16-none.toml --tag bf16
The P-patch pass is the single-patch `GlobalStep.step` with P query rows: positions pos .. pos + P - 1, a P-row causal mask,
rotary embeddings for P positions, and P cache writes. Output: results/drafting/stage4/global_multi-<tag>.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from byteprof.config import load_config  # noqa: E402
from byteprof.runner import generate_checked, load_model, load_prompts, set_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=600)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--max-p", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config, {"max_new_bytes": 64})
    model, tok = load_model(cfg)
    from byteprof import optim

    handle = optim.apply(model, tok, cfg)
    glob, loop = handle.glob, handle.loop
    sync = torch.cuda.synchronize
    set_seed(cfg.seed)
    generate_checked(model, tok, load_prompts(ROOT / "prompts/default.jsonl")[0]["text"], cfg, sync)
    n0 = int(glob.pos.item())
    m = glob.m
    d = loop.d
    dev = cfg.device
    dtype = loop.h_byte.dtype
    hin = torch.randn(1, a.max_p, d, device=dev, dtype=dtype)
    out_buf = torch.zeros(1, a.max_p, d, device=dev, dtype=dtype)
    ar = torch.arange(a.max_p, device=dev)

    def step(p: int) -> None:
        cp = glob.pos + ar[:p]
        position_ids = cp.unsqueeze(0)
        pe = m.rotary_embs["full_attention"](hin[:, :p], position_ids)
        mask = (glob.kv_index[None, :] <= cp[:, None]).view(1, 1, p, -1)
        h = hin[:, :p]
        for layer in m.layers:
            h = layer(h, attention_mask=mask, position_ids=position_ids, past_key_values=glob.cache,
                      cache_position=cp, position_embeddings=pe)
        out_buf[:, :p].copy_(h)

    res: dict = {"config": a.config, "n0": n0}
    graphs = {}
    with torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for p in range(1, a.max_p + 1):
                    step(p)
        torch.cuda.current_stream().wait_stream(s)
        sync()
        for p in range(1, a.max_p + 1):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step(p)
            graphs[p] = g
        sync()
        graphs_single = handle.runner.graphs["glob"]
        for name, g in [("single (existing graph)", graphs_single)] + [(f"P={p}", graphs[p]) for p in graphs]:
            trials = []
            for _ in range(a.trials):
                glob.pos.fill_(n0)
                sync()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for r in range(a.reps):
                    if r % 100 == 0:
                        glob.pos.fill_(n0)
                    g.replay()
                e1.record()
                sync()
                trials.append(e0.elapsed_time(e1) / a.reps)
            res[name] = {"ms": statistics.median(trials), "trials": trials}
            print(name, f"{res[name]['ms']:.4f} ms", flush=True)
    out = ROOT / "results" / "drafting" / "stage4"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"global_multi-{a.tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
