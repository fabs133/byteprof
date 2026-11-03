"""Stage 0 (TASK_build_p4.md): GPU time of one crossing round, by K and P, against what the cost model already has.

The crossing round = encd{K}, gx{P} (pool P patches + the global layers over P rows), decx{K} (multi-patch decoder, raw-id
acceptance, lazy commit, rollback), replayed back to back with CUDA events (reps per trial, median of trials). L = 0 (no
draft matches, as in the Task A verify bench `*-fused-L0`). Compared with

  * pair(K)  = encd{K} + decd{K}n  (Task A's "round"), measured in the same session, interleaved trial by trial
  * B(P)     = the 16 global layers over P rows without pooling (draft_global_multi_bench.py's graph), same session
  * dp       = dec_p - dec_n at K = 0 (the new-patch cost of the decoder), same session
  * the stored cost-model inputs (results/parallel/bench/<prec>-fused-L0, results/drafting/stage4/global_multi-<prec>-p8)

extra(K, P) = X(K, P) - (pair + B(P) + dp): the number H48 tests.

Usage: python tools/p4_cross_bench.py --config configs/p4/bf16-bench.toml --prec bf16 --ks 8,16,32 [--reps 150 --trials 5]
Output: results/p4/stage0/<prec>-k<ks>.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from byteprof.config import load_config  # noqa: E402
from byteprof.runner import generate_checked, load_model, load_prompts, set_seed  # noqa: E402


def interleaved(fns: dict, reps: int, trials: int, pre=None, pre_every: int = 50) -> dict:
    """ms per replay of each callable: events around `reps` back-to-back calls, trial by trial in turn."""
    out = {k: [] for k in fns}
    for _ in range(trials):
        for name, fn in fns.items():
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for r in range(reps):
                if pre is not None and r % pre_every == 0:
                    pre()
                fn()
            b.record()
            torch.cuda.synchronize()
            out[name].append(a.elapsed_time(b) / reps)
    return {k: {"ms": statistics.median(v), "trials": v} for k, v in out.items()}


def flag_pattern(k: int, p: int) -> list[int]:
    """Flags of the K + 1 inputs: a0 and p - 1 further positions spread over 1..K."""
    flags = [0] * (k + 1)
    flags[0] = 1
    pos = sorted({int(round(x)) for x in np.linspace(1, k, p - 1)}) if p > 1 else []
    assert len(pos) == p - 1, (k, p, pos)
    for j in pos:
        flags[j] = 1
    return flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prec", required=True)
    ap.add_argument("--ks", default="8,16,32")
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", default="results/p4/stage0")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    cfg = load_config(ROOT / a.config, {"max_new_bytes": 96})
    model, tok = load_model(cfg)
    from byteprof import optim
    from byteprof.optim import draft_gated as gated

    t0 = time.time()
    handle = optim.apply(model, tok, cfg)
    print(f"apply {time.time() - t0:.1f}s", flush=True)
    runner, glob, loop, engine = handle.runner, handle.glob, handle.loop, handle.drafting["engine"]
    g = runner.graphs
    sync = torch.cuda.synchronize
    set_seed(cfg.seed)
    generate_checked(model, tok, load_prompts(ROOT / "prompts/default.jsonl")[0]["text"], cfg, sync)
    n0 = int(glob.pos.item())
    d = loop.d
    dev = cfg.device
    m = glob.m
    ar = torch.arange(8, device=dev)
    hin = torch.randn(1, 8, d, device=dev, dtype=loop.h_byte.dtype)
    out_buf = torch.zeros(1, 8, d, device=dev, dtype=loop.h_byte.dtype)

    def bstep(p: int) -> None:  # the global layers over p rows, no pooling (the cost model's B(P))
        cp = glob.pos + ar[:p]
        position_ids = cp.unsqueeze(0)
        pe = m.rotary_embs["full_attention"](hin[:, :p], position_ids)
        mask = (glob.kv_index[None, :] <= cp[:, None]).view(1, 1, p, -1)
        h = hin[:, :p]
        for layer in m.layers:
            h = layer(h, attention_mask=mask, position_ids=position_ids, past_key_values=glob.cache, cache_position=cp,
                      position_embeddings=pe)
        out_buf[:, :p].copy_(h)

    bgraphs = {}
    with torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for p in range(1, 9):
                    bstep(p)
        torch.cuda.current_stream().wait_stream(s)
        sync()
        for p in range(1, 9):
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                bstep(p)
            bgraphs[p] = gr
        sync()

        stored = json.loads((ROOT / f"results/parallel/bench/{a.prec}-fused-L0/bench.json").read_text(encoding="utf-8"))
        stored_pair = {int(k): v["ms"] for k, v in stored["rounds"].items()}
        stored_dp = stored["pieces"]["Cp|0"]["ms"] - stored["pieces"]["Cn|0"]["ms"]
        gm = json.loads((ROOT / f"results/drafting/stage4/global_multi-{a.prec}-p8.json").read_text(encoding="utf-8"))
        stored_b = {p: gm[f"P={p}"]["ms"] for p in range(1, 9)}

        def reset_pos():
            glob.pos.fill_(n0)

        res: dict = {"config": a.config, "prec": a.prec, "reps": a.reps, "trials": a.trials, "n0": n0,
                     "stored": {"pair": stored_pair, "dp": stored_dp, "B": stored_b}, "cells": {}}
        smax = engine.smax
        rng = np.random.default_rng(0)
        for k in ks:
            s_ = k + 1
            for p in range(1, min(8, k + 1) + 1):
                flags = flag_pattern(k, p)
                psel, pool = gated.patch_select(flags), gated.pool_positions(flags)
                dn = engine.d_np
                dn[:] = 0
                dn[:s_] = rng.integers(4, 260, s_)
                dn[smax:smax + s_] = 1
                dn[engine.o_raw:engine.o_raw + s_] = 3  # never predicted: L = 0
                dn[engine.o_psel:engine.o_psel + s_] = psel
                dn[engine.o_pool:engine.o_pool + p] = pool
                engine.d_ids.copy_(engine.p_d)
                sync()

                def gr(name):
                    return g[name].replay

                enc, gx, decx, glob1 = gr(f"encd{k}"), gr(f"gx{p}"), gr(f"decx{k}"), gr("glob")
                decn, decp = gr(f"decd{k}n"), gr(f"decd{k}p")

                def X():
                    enc(); gx(); decx()

                def Y():  # the existing single-patch boundary round
                    enc(); glob1(); decp()

                def pair():
                    enc(); decn()

                fns = {"X": X, "Y": Y, "pair": pair, "Bnow": bgraphs[p].replay, "Gx": gx, "decx": decx, "Cp": decp,
                       "Cn": decn, "dec_p0": gr("dec_p"), "dec_n0": gr("dec_n")}
                r = interleaved(fns, a.reps, a.trials, pre=reset_pos)
                reset_pos()
                enc(); gx(); decx()
                sync()
                assert int(engine.out_dev[smax]) == 0, "L != 0"
                x = r["X"]["ms"]
                dp_now = r["dec_p0"]["ms"] - r["dec_n0"]["ms"]
                extra_now = x - (r["pair"]["ms"] + r["Bnow"]["ms"] + dp_now)
                extra_stored = x - (stored_pair[k] + stored_b[p] + stored_dp)
                res["cells"][f"{k}|{p}"] = {**{kk: v["ms"] for kk, v in r.items()}, "trials": {kk: v["trials"] for kk, v in r.items()},
                                           "dp_now": dp_now, "extra_now": extra_now, "extra_stored": extra_stored,
                                           "pool": pool, "psel": psel}
                print(f"K={k:<2} P={p}: X {x:.3f}  pair {r['pair']['ms']:.3f}  B {r['Bnow']['ms']:.3f}  dp {dp_now:.3f}  "
                      f"extra(now) {extra_now:+.3f}  extra(stored) {extra_stored:+.3f}  | Y {r['Y']['ms']:.3f}  Gx {r['Gx']['ms']:.3f}  "
                      f"decx {r['decx']['ms']:.3f} Cp {r['Cp']['ms']:.3f}", flush=True)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.prec}-k{'-'.join(map(str, ks))}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
