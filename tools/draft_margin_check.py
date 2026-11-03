"""Are the bf16 / int4 divergences of the drafting loop near-ties? For the prompts that diverge, record the top-2 logit gap of the
non-drafted model (eager path, bit-identical to the minimal loop) at the step where the drafted output first differs, and compare it
with the distribution of gaps over all steps.

Usage: python tools/draft_margin_check.py --config configs/draft/bolmo-bf16-none.toml --exact results/drafting/exact/bf16-combined-k6.json
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
from byteprof.runner import generate_checked, load_model, load_prompts  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exact", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = json.loads((ROOT / a.exact).read_text(encoding="utf-8"))["rows"]
    div = {}
    for r in rows:
        if r["first_divergence"] is not None and r["n"] == 256:
            div[r["id"]] = r["first_divergence"]
    # prompts that only diverge in the 1024-byte run
    for r in rows:
        if r["first_divergence"] is not None and r["id"] not in div:
            div[r["id"]] = r["first_divergence"]
    cfg = load_config(ROOT / a.config, {"max_new_bytes": 1100})
    model, tok = load_model(cfg)
    prompts = {p["id"]: p["text"] for f in ("prompts/default.jsonl", "prompts/drafting_ext.jsonl") for p in load_prompts(ROOT / f)}
    gaps: list[float] = []
    hook = model.lm_head.register_forward_hook(lambda m, i, o: gaps.append(
        float((lambda t: t[0] - t[1])(torch.topk(o[0, -1].float(), 2).values))))
    res = {}
    for pid, d in div.items():
        gaps.clear()
        generate_checked(model, tok, prompts[pid], cfg, torch.cuda.synchronize)
        # call 0 is the prefill (a_0); the atom a_k is predicted by call k; new byte index d is atom a_{d+1}
        at = gaps[d + 1] if d + 1 < len(gaps) else None
        allg = gaps[1:d + 1]
        res[pid] = {"first_divergence": d, "gap_at_divergence": at, "median_gap_before": statistics.median(allg) if allg else None,
                    "share_of_earlier_steps_with_smaller_gap": (sum(1 for g in allg if g <= at) / len(allg)) if (allg and at is not None) else None}
        print(pid, res[pid], flush=True)
    hook.remove()
    if a.out:
        (ROOT / a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
