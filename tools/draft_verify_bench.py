"""Stage 1: verify-cost microbenchmark. Replays every graphed piece 1,000 times (CUDA events), median of 5 trials.

Pieces: A_K = encd{K} (K = 0: enc_m), C_K^n / C_K^p = decd{K}n / decd{K}p (K = 0: dec_n / dec_p), B = glob.
A "round" = A_K then C_K^n replayed back to back (no new patch), the unit of H23.
Usage: python tools/draft_verify_bench.py --config configs/draft/bolmo-bf16-context.toml --tag bf16
Output: results/drafting/stage1/<tag>/bench.json, bench.md
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


def time_graph(fn, reps: int, trials: int, pre=None, pre_every: int = 0) -> tuple[float, list[float]]:
    """ms per replay: events around `reps` back-to-back replays; `pre` runs every `pre_every` replays."""
    out = []
    for _ in range(trials):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for r in range(reps):
            if pre is not None and pre_every and r % pre_every == 0:
                pre()
            fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / reps)
    return statistics.median(out), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--root", default="results/drafting/stage1")
    ap.add_argument("--accept", choices=["asis", "none"], default="asis",
                    help="asis: whatever the drafts left in the buffer give; none: L = 0 (no draft ever matches)")
    ap.add_argument("--trials", type=int, default=5)
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config, {"max_new_bytes": 96})
    model, tok = load_model(cfg)
    from byteprof import optim

    handle = optim.apply(model, tok, cfg)
    runner, glob, engine = handle.runner, handle.glob, handle.drafting["engine"]
    sync = torch.cuda.synchronize
    set_seed(cfg.seed)
    prompt = load_prompts(ROOT / "prompts/default.jsonl")[0]["text"]
    generate_checked(model, tok, prompt, cfg, sync)  # puts valid states into the static buffers
    n0 = int(glob.pos.item())
    g = runner.graphs
    ks = engine.k_values

    def rep(name):
        return g[name].replay

    def reset_pos():
        glob.pos.fill_(n0)

    res: dict = {"config": a.config, "reps": a.reps, "trials": a.trials, "pieces": {}, "rounds": {}}
    with torch.inference_mode():
        for k in ks:
            names = {"A": "enc_m" if k == 0 else f"encd{k}", "Cn": "dec_n" if k == 0 else f"decd{k}n",
                     "Cp": "dec_p" if k == 0 else f"decd{k}p"}
            for key, name in names.items():
                med, all_ = time_graph(rep(name), a.reps, a.trials)
                res["pieces"][f"{key}|{k}"] = {"ms": med, "trials": all_, "graph": name}
            ea, ec = rep(names["A"]), rep(names["Cn"])
            if k > 0 and a.accept != "asis":
                s_ = k + 1
                engine.d_ids[1:s_].fill_(3)  # <bpe_token_end>: a valid embedding index the model never predicts
                ea()
                ec()
                assert int(engine.out_dev[engine.smax]) == 0

            def pair(ea=ea, ec=ec):
                ea()
                ec()

            med, all_ = time_graph(pair, a.reps, a.trials)
            res["rounds"][str(k)] = {"ms": med, "trials": all_}
        glob.pos.fill_(n0)
        med, all_ = time_graph(rep("glob"), a.reps, a.trials, pre=reset_pos, pre_every=300)
        res["pieces"]["B"] = {"ms": med, "trials": all_, "graph": "glob"}
    base = res["rounds"]["0"]["ms"]
    for k in ks:
        res["rounds"][str(k)]["ratio_to_k0"] = res["rounds"][str(k)]["ms"] / base
    out = ROOT / a.root / a.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = [f"# Stage 1 verify cost, {a.tag} ({a.config})", "",
         f"GPU ms per replay, median of {a.trials} trials of {a.reps} back-to-back replays.", "",
         "| K | A_K | C_K (no patch) | C_K (new patch) | A_K + C_K replayed as a round | ratio to K = 0 |",
         "|---:|---:|---:|---:|---:|---:|"]
    for k in ks:
        p = res["pieces"]
        L.append(f"| {k} | {p[f'A|{k}']['ms']:.4f} | {p[f'Cn|{k}']['ms']:.4f} | {p[f'Cp|{k}']['ms']:.4f} | "
                 f"{res['rounds'][str(k)]['ms']:.4f} | {res['rounds'][str(k)]['ratio_to_k0']:.3f} |")
    L += ["", f"B (global graph, one patch): {res['pieces']['B']['ms']:.4f} ms"]
    (out / "bench.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
