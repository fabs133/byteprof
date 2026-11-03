"""OLMo 2 ground truth for the token drafters: the existing minimal graph loop on the default + extended prompts.

Usage: python tools/olmo_truth.py --config configs/fair/olmo-bf16.toml --tag olmo-bf16 --lengths 72,288
Output: results/drafting/truth/<tag>.json = {prompts: {id: {set, type, ids (prompt token ids), tokens: {n: [t0 .. t_{n-1}]},
        bytes: {n: UTF-8 bytes of the n tokens}}}}  (t0 is the first token, produced by the prefill)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from byteprof.config import load_config  # noqa: E402
from byteprof.runner import load_model, load_prompts  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lengths", default="72,288")
    ap.add_argument("--prompts", default="prompts/default.jsonl,prompts/drafting_ext.jsonl")
    a = ap.parse_args()
    lengths = sorted(int(x) for x in a.lengths.split(","))
    cfg = load_config(ROOT / a.config)
    model, tok = load_model(cfg)
    from byteprof import optim

    handle = optim.apply(model, tok, cfg)
    res = {"config": a.config, "tag": a.tag, "lengths": lengths, "prompts": {}}
    for f in a.prompts.split(","):
        for p in load_prompts(ROOT / f):
            ids = tok(p["text"], return_tensors="pt")["input_ids"].to(cfg.device)
            rec = {"set": Path(f).stem, "text": p["text"], "ids": ids[0].tolist(), "tokens": {}, "bytes": {}}
            for n in lengths:
                with torch.inference_mode():
                    out = model.generate(ids, max_new_tokens=n, do_sample=False)
                new = out[0, ids.shape[1]:].tolist()
                rec["tokens"][str(n)] = new
                rec["bytes"][str(n)] = len(tok.decode(new, skip_special_tokens=True).encode("utf-8"))
            res["prompts"][p["id"]] = rec
            print(p["id"], {n: (len(v), rec["bytes"][n]) for n, v in rec["tokens"].items()}, flush=True)
    out_dir = ROOT / "results" / "drafting" / "truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{a.tag}.json").write_text(json.dumps(res), encoding="utf-8")
    handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
