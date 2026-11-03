"""Stage A0 step 3: time of every distinct local-stack linear at M = 1, 2, 5, 9, 17, 33 rows (CUDA graph of 20 calls,
median of 5 trials of 200 replays). Usage: python tools/linear_m_scaling.py --config <draft config> --tag int4
Output: results/parallel/linear_m_<tag>.json / .md"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from byteprof.config import load_config  # noqa: E402
from byteprof.runner import load_model  # noqa: E402

MS = (1, 2, 5, 9, 17, 33)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config, {"max_new_bytes": 32})
    model, tok = load_model(cfg)
    from byteprof import optim

    handle = optim.apply(model, tok, cfg)
    bm = model.model
    mods = {}
    for part, root in (("enc", bm.local_encoder), ("dec", bm.local_decoder)):
        for li, layer in enumerate(root.layers):
            for name, m in layer.named_modules():
                if isinstance(m, torch.nn.Linear):
                    mods[f"{part}{li}.{name}"] = m
    groups: dict[tuple, list[str]] = {}
    for n, m in mods.items():
        groups.setdefault((m.in_features, m.out_features, type(m.weight).__name__), []).append(n)
    res = {}
    with torch.inference_mode():
        for (i, o, wt), names in groups.items():
            lin = mods[names[0]]
            row = {}
            for m_ in MS:
                x = torch.randn(m_, i, device="cuda", dtype=torch.bfloat16)
                for _ in range(5):
                    lin(x)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(20):
                        lin(x)
                trials = []
                for _ in range(5):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    e0.record()
                    for _ in range(200):
                        g.replay()
                    e1.record()
                    torch.cuda.synchronize()
                    trials.append(e0.elapsed_time(e1) / 200 / 20 * 1000)
                row[m_] = statistics.median(trials)
            res[f"{i}x{o} {wt}"] = {"count": len(names), "us": row, "names": names[:3]}
    out = ROOT / "results" / "parallel"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"linear_m_{a.tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = [f"| linear (in x out, weight type) | count | " + " | ".join(f"M={m}" for m in MS) + " |", "|---|---:|" + "---:|" * len(MS)]
    tot = {m: 0.0 for m in MS}
    for k, v in res.items():
        L.append(f"| {k} | {v['count']} | " + " | ".join(f"{v['us'][m]:.2f}" for m in MS) + " |")
        for m in MS:
            tot[m] += v["us"][m] * v["count"]
    L.append("| **sum over the local stack (us)** | | " + " | ".join(f"{tot[m]:.1f}" for m in MS) + " |")
    (out / f"linear_m_{a.tag}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
