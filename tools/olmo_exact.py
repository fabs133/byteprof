"""Exactness of OLMo 2 token drafting against the recorded non-drafted minimal graph loop: first diverging token per prompt/length.

Usage: python tools/olmo_exact.py --config configs/draft/olmo-fp32-combined-k8.toml --truth results/drafting/truth/olmo-fp32.json --out results/drafting/exact/olmo-fp32-combined-k8.json
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from byteprof.config import load_config  # noqa: E402
from byteprof.runner import load_model, load_prompts  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    truth = json.loads((ROOT / a.truth).read_text(encoding="utf-8"))["prompts"]
    model, tok = load_model(cfg)
    from byteprof import optim

    handle = optim.apply(model, tok, cfg)
    stats = handle.stats
    rows = []
    for pid, rec in truth.items():
        ids = torch.tensor([rec["ids"]], device=cfg.device)
        for n, ref in rec["tokens"].items():
            stats.__init__()
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=int(n), do_sample=False)
            got = out[0, ids.shape[1]:].tolist()
            m = min(len(got), len(ref))
            div = next((i for i in range(m) if got[i] != ref[i]), None if len(got) == len(ref) else m)
            d = stats.dump()
            rows.append({"id": pid, "n": int(n), "tokens": len(ref), "first_divergence": div, "rounds": d["rounds"],
                         "tokens_per_round": (len(ref) - 1) / max(d["rounds"], 1), "by_k": d["by_k"]})
            print(f"{pid:<12} n={n:<4} tokens={len(ref):<4} div={div} rounds={d['rounds']} tok/round={rows[-1]['tokens_per_round']:.2f}", flush=True)
    bad = [r for r in rows if r["first_divergence"] is not None]
    print(f"{len(rows) - len(bad)}/{len(rows)} identical; diverging: {[(r['id'], r['n'], r['first_divergence']) for r in bad]}")
    if a.out:
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"config": a.config, "truth": a.truth, "rows": rows}, indent=1), encoding="utf-8")
    handle.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
