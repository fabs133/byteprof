"""Exactness of the drafting loop against the recorded non-drafted minimal loop (H26): first divergence of the emitted
atoms (byte AND boundary flag) per prompt and length.

Usage: python tools/draft_exact.py --config configs/draft/bolmo-fp32-combined.toml --truth results/drafting/truth/fp32.json
       [--lengths 256,1024] [--prompts prompts/default.jsonl,prompts/drafting_ext.jsonl] [--out results/drafting/exact/fp32-combined.json]
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from byteprof.config import load_config  # noqa: E402
from byteprof.runner import _MaxAtoms, load_model, load_prompts  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--lengths", default="256,1024")
    ap.add_argument("--prompts", default="prompts/default.jsonl,prompts/drafting_ext.jsonl")
    ap.add_argument("--out")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]
    cfg = load_config(ROOT / a.config)
    truth = json.loads((ROOT / a.truth).read_text(encoding="utf-8"))["prompts"]
    prompts = [p for f in a.prompts.split(",") for p in load_prompts(ROOT / f)]
    model, tok = load_model(cfg)
    from byteprof import optim
    from transformers import StoppingCriteriaList

    handle = optim.apply(model, tok, cfg)
    d = handle.drafting
    rec, st = d["rec"], d["stats"]
    rows = []
    for p in prompts:
        ids = tok(p["text"], return_tensors="pt")["input_ids"].to(cfg.device)
        for n in lengths:
            ref = truth[p["id"]]["raw"][str(n)]  # a_0 .. a_m (m may be < n if the recorder stopped early)
            m = len(ref) - 1
            rec["raws"].clear()
            rec["on"] = True
            st.__init__()
            t0 = time.perf_counter()
            cache_full = False
            try:
                with torch.inference_mode():
                    model.generate(ids, max_new_tokens=m, stopping_criteria=StoppingCriteriaList([_MaxAtoms(m)]))
            except RuntimeError as exc:  # a diverged (denser-patch) text can fill the 512-patch static cache before m atoms
                if "static global cache is full" not in str(exc):
                    raise
                cache_full = True
            ms = (time.perf_counter() - t0) * 1000
            rec["on"] = False
            got = rec["raws"]
            k = min(len(got), m)
            div = next((i for i in range(k) if got[i] != ref[1 + i]), None if (len(got) == m or cache_full) else k)
            dump = st.dump()
            rows.append({"id": p["id"], "n": n, "atoms": m, "emitted": len(got), "first_divergence": div,
                         "rounds": dump["rounds"], "bytes_per_round": dump["bytes_per_round"], "by_k": dump["by_k"],
                         "ms": ms, "cache_full": cache_full})
            print(f"{p['id']:<12} n={n:<5} atoms={m:<5} div={div}{' (cache full after %d atoms)' % len(got) if cache_full else ''} rounds={dump['rounds']} B/round={dump['bytes_per_round']:.2f}", flush=True)
    bad = [r for r in rows if r["first_divergence"] is not None]
    print(f"{len(rows) - len(bad)}/{len(rows)} identical; diverging: {[(r['id'], r['n'], r['first_divergence']) for r in bad]}")
    if a.out:
        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"config": a.config, "truth": a.truth, "rows": rows}, indent=1), encoding="utf-8")
    handle.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
