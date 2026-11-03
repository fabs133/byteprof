"""Self-consistent oracle truth for one drafting configuration (TASK_drafting.md, oracle = upper bound only).

In bf16 / int4 the batched verification rounds differ numerically from the M = 1 minimal loop, so the minimal loop's output
is not what the drafting loop produces after its first divergence, and an oracle fed with it drafts garbage from there on.
This tool iterates: run the oracle drafter with truth T_i, record the loop's own emitted atoms, use them as T_{i+1}, until
T_{i+1} == T_i (a fixed point: the oracle proposes exactly what the loop then emits) or the iteration limit is reached.

Usage: python tools/draft_oracle_truth.py --config configs/draft/bolmo-bf16-oracle-k6.toml --out results/drafting/truth/oracle-bf16-k6.json
Only the prompts of `--prompts` (default prompts) at `--n` new bytes. Output has the layout of draft_truth.py.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from byteprof.config import load_config  # noqa: E402
from byteprof.drafting.truthio import raw_to_atom  # noqa: E402
from byteprof.runner import _MaxAtoms, load_model, load_prompts  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="prompts/default.jsonl")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--max-iter", type=int, default=10)
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    seed = json.loads((ROOT / cfg.draft["truth_file"]).read_text(encoding="utf-8"))["prompts"]
    prompts = [p for f in a.prompts.split(",") for p in load_prompts(ROOT / f)]
    model, tok = load_model(cfg)
    from byteprof import optim
    from transformers import StoppingCriteriaList

    handle = optim.apply(model, tok, cfg)
    d = handle.drafting
    oracle, rec = d["oracle"], d["rec"]
    cur = {p["id"]: [r for r in seed[p["id"]]["raw"][str(a.n)]] for p in prompts}  # raw ids a_0 .. a_n
    ids_of = {p["id"]: tuple(seed[p["id"]]["ids"]) for p in prompts}
    history = []
    for it in range(1, a.max_iter + 1):
        changed = []
        for p in prompts:
            oracle[ids_of[p["id"]]] = [x for x in (raw_to_atom(r) for r in cur[p["id"]]) if x is not None]
            ids = torch.tensor([list(ids_of[p["id"]])], device=cfg.device)
            rec["raws"].clear()
            rec["on"] = True
            with torch.inference_mode():
                model.generate(ids, max_new_tokens=a.n, stopping_criteria=StoppingCriteriaList([_MaxAtoms(a.n)]))
            rec["on"] = False
            new = [cur[p["id"]][0]] + list(rec["raws"])
            if new != cur[p["id"]]:
                changed.append((p["id"], next((i for i, (x, y) in enumerate(zip(new, cur[p["id"]])) if x != y), min(len(new), len(cur[p["id"]])))))
            cur[p["id"]] = new
        history.append(changed)
        print(f"iteration {it}: {len(changed)} prompt(s) changed {changed}", flush=True)
        if not changed:
            break
    out = {"config": a.config, "fixed_point": not history[-1], "iterations": len(history), "history": history,
           "prompts": {pid: {"set": "default", "ids": list(ids_of[pid]), "raw": {str(a.n): cur[pid]}} for pid in cur}}
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out), encoding="utf-8")
    handle.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
