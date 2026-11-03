"""Stage C5 of TASK_fair_ratio_floor.md: cost of the per-byte sync, by differencing replays of piece C.

Loads the Bolmo variant of the given config (graph-global + minimal loop, as in the speed runs), then
times, per graph (enc_m, dec_n, dec_p, glob), 1000 back-to-back replays:
  (a) no sync between replays (one sync at the end);
  (b) pinned-memory read after each replay: device-to-host copy of the token, event record, synchronize.
The difference (b - a) per replay is the cost of the sync plus its memcpy. Also (c): GPU time of one
replay by CUDA events (no host involvement), to compare with (a).
Run: python -X utf8 tools/sync_cost_check.py --config configs/fair/bolmo-all-int4.toml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from byteprof import optim, runner  # noqa: E402
from byteprof.config import load_config  # noqa: E402

N = 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/fair/sync_cost.json")
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    model, tok = runner.load_model(cfg)
    handle = optim.apply(model, tok, cfg)
    rn = handle.runner
    pieces = handle.pieces
    pinned = torch.zeros(1, dtype=torch.long).pin_memory()
    done = torch.cuda.Event()
    res = {"config": args.config, "replays": N, "graphs": {}}
    # state buffers hold zeros / junk; timing does not depend on values
    for name in ("enc_m", "dec_n", "dec_p", "glob"):
        g = rn.graphs[name]
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        rows = {}
        n = 300 if name == "glob" else N  # glob advances its static KV position (512 patches) per replay
        for rep in range(3):
            pieces.glob.pos.zero_()
            t0 = time.perf_counter()
            for _ in range(n):
                g.replay()
            torch.cuda.synchronize()
            pieces.glob.pos.zero_()
            a = (time.perf_counter() - t0) / n * 1e6
            t0 = time.perf_counter()
            for _ in range(n):
                g.replay()
                pinned.copy_(pieces.tok, non_blocking=True)
                done.record()
                done.synchronize()
            b = (time.perf_counter() - t0) / n * 1e6
            pieces.glob.pos.zero_()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(n):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            c = e0.elapsed_time(e1) * 1000 / n
            rows[rep] = {"no_sync_us": a, "with_sync_us": b, "sync_cost_us": b - a, "gpu_event_us": c}
        res["graphs"][name] = rows
        best = min(rows.values(), key=lambda r: r["sync_cost_us"])
        print(f"{name:6s}: back-to-back {rows[2]['no_sync_us']:.1f} us/replay, with pinned read+sync "
              f"{rows[2]['with_sync_us']:.1f} us, sync cost {rows[2]['sync_cost_us']:.1f} us "
              f"(min over 3: {best['sync_cost_us']:.1f}), GPU events {rows[2]['gpu_event_us']:.1f} us", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
