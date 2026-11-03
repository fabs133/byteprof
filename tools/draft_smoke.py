"""Quick check of the drafting loop against the recorded minimal-loop truth: first divergence per prompt, round stats.

Usage: python tools/draft_smoke.py --config configs/draft/bolmo-bf16-combined.toml --truth results/drafting/truth/bf16.json --n 256
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from byteprof.config import load_config  # noqa: E402
from byteprof.drafting.truthio import raw_to_atom  # noqa: E402
from byteprof.runner import generate_checked, load_model, load_prompts, set_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--truth", default="results/drafting/truth/bf16.json")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--prompts", default="prompts/default.jsonl")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config, {"max_new_bytes": a.n})
    truth = json.loads((ROOT / a.truth).read_text(encoding="utf-8"))["prompts"]
    prompts = [p for f in a.prompts.split(",") for p in load_prompts(ROOT / f)]
    model, tok = load_model(cfg)
    from byteprof import optim
    t0 = time.time()
    handle = optim.apply(model, tok, cfg)
    print(f"apply {time.time() - t0:.1f}s; graphs: {sorted(handle.runner.graphs)}", flush=True)
    st = handle.drafting["stats"]
    sync = torch.cuda.synchronize
    set_seed(cfg.seed)
    tot_n = tot_ms = 0
    for p in prompts:
        st.__init__()
        r = generate_checked(model, tok, p["text"], cfg, sync, want_text=True)
        ref = truth[p["id"]]["raw"][str(a.n)]
        ref_b = bytes(raw_to_atom(x) & 255 for x in ref[1:] if raw_to_atom(x) is not None)
        got = r.text
        m = min(len(got), len(ref_b))
        div = next((i for i in range(m) if got[i] != ref_b[i]), None if len(got) == len(ref_b) else m)
        d = st.dump()
        print(f"{p['id']:<12} bytes={r.new_bytes} {r.ms:.0f}ms {r.ms / max(r.new_bytes, 1):.3f} ms/B div={div} "
              f"rounds={d['rounds']} B/round={d['bytes_per_round']:.2f} byK={d['by_k']}", flush=True)
        tot_n += r.new_bytes; tot_ms += r.ms
    print(f"overall {tot_ms / tot_n:.3f} ms/byte (single pass, includes first-call effects)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
