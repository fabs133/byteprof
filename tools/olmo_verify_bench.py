"""OLMo 2 verify cost: replays the graph that verifies K + 1 tokens (K = 0 is the plain one-token step).

Usage: python tools/olmo_verify_bench.py --config configs/draft/olmo-bf16-context-k8.toml --tag olmo-bf16
The graphs advance the cache position by L + 1 per replay (random drafts: L = 0), so the state is re-initialized every 100
replays. Output: results/drafting/stage3/olmo_bench-<tag>.json
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
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config, {"max_new_bytes": 24})
    model, tok = load_model(cfg)
    from byteprof import optim

    handle = optim.apply(model, tok, cfg)
    sync = torch.cuda.synchronize
    set_seed(cfg.seed)
    prompt = load_prompts(ROOT / "prompts/default.jsonl")[0]["text"]
    generate_checked(model, tok, prompt, cfg, sync)
    st = handle.draft_state
    n_prompt = len(tok(prompt)["input_ids"])
    base = torch.tensor(n_prompt + 1, device=cfg.device)
    res = {}
    with torch.inference_mode():
        for k, g in sorted(st["graphs"].items()):
            trials = []
            for _ in range(a.trials):
                sync()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for r in range(a.reps):
                    if r % 100 == 0:
                        st["set_state"](None, base)
                    g.replay()
                e1.record()
                sync()
                trials.append(e0.elapsed_time(e1) / a.reps)
            res[k] = {"ms": statistics.median(trials), "trials": trials}
            print(k, f"{res[k]['ms']:.4f} ms", flush=True)
    for k in res:
        res[k]["ratio_to_k0"] = res[k]["ms"] / res[0]["ms"]
    out = ROOT / "results" / "drafting" / "stage3"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"olmo_bench-{a.tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
