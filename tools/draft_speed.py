"""Stage 3 speed cell: one config, same protocol as `byteprof profile` for optimized variants (same prompts, seed,
2 warmup passes, measured_runs x prompts generations, ms/byte = wall / new bytes incl. prefill), plus the drafting
statistics (rounds, emitted bytes per round, K mix) of every generation.

Usage: python tools/draft_speed.py --config configs/draft/bolmo-bf16-combined-k6.toml --out results/drafting/speed/bf16/combined-k6-run1.json
Works for Bolmo drafting configs, Bolmo `kind = none` (the minimal loop) and, with the same flags, OLMo configs.
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
from byteprof.runner import generate_checked, load_model, load_prompts, set_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int)
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config, {"measured_runs": a.runs} if a.runs else None)
    prompts = load_prompts(ROOT / cfg.prompts_file)
    use_cuda = cfg.device.startswith("cuda")
    sync = torch.cuda.synchronize
    t_start = time.time()
    model, tok = load_model(cfg)
    from byteprof import optim

    handle = optim.apply(model, tok, cfg)
    st = getattr(handle, "drafting", None)
    stats = st["stats"] if st else getattr(handle, "stats", None)
    warm_passes = max(2, -(-cfg.warmup_runs // len(prompts)))
    set_seed(cfg.seed)
    for _ in range(warm_passes):
        for p in prompts:
            generate_checked(model, tok, p["text"], cfg, sync, extra_kwargs=handle.generate_kwargs())
    torch.cuda.reset_peak_memory_stats()
    set_seed(cfg.seed)
    rows = []
    for _ in range(cfg.measured_runs):
        for p in prompts:
            if stats is not None:
                stats.__init__()
            r = generate_checked(model, tok, p["text"], cfg, sync, want_text=True, extra_kwargs=handle.generate_kwargs())
            d = stats.dump() if stats is not None else {}
            rows.append({"prompt": p["id"], "new_bytes": r.new_bytes, "ms": r.ms, "text": r.text.decode("utf-8", "replace"),
                         **{k: d[k] for k in ("rounds", "emitted", "by_k", "l_hist", "global_calls", "kinds", "kind_emitted", "patches_hist",
                                  "patches_kept", "global_patches", "cap_cut", "retok_us_per_call", "retok_calls",
                                  "propose_us_per_round") if k in d}})
    nb, ms = sum(r["new_bytes"] for r in rows), sum(r["ms"] for r in rows)
    out = {"config": a.config, "ms_per_byte": ms / nb, "bytes_per_s": nb / ms * 1000, "new_bytes": nb, "wall_ms": ms,
           "peak_mib": torch.cuda.max_memory_allocated() / 2**20 if use_cuda else 0.0, "rows": rows,
           "rounds_per_byte": (sum(r.get("rounds", 0) for r in rows) / nb) if rows and "rounds" in rows[0] else None,
           "details": handle.describe(), "seconds": time.time() - t_start,
           "draft": dict(cfg.draft), "quantize": dict(cfg.quantize)}
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{a.config}: {out['ms_per_byte']:.3f} ms/byte  {out['bytes_per_s']:.0f} B/s  rounds/byte={out['rounds_per_byte']}")
    handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
