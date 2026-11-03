"""Step-by-step comparison of the minimal Bolmo decode loop against the original generate().

For one prompt: run the original (eager) generate() with a forward hook on BolmoModel that
snapshots, after every model step, the recurrent state of all five local layers, the decoder's
last_value and the subword ids passed in; then install the optimization (minimal loop) and run
again with a debug callback that snapshots the static buffers after every byte. Reports the
first byte where anything differs.

    python tools/debug_minimal_loop.py --config configs/bolmo-1b-minimal.toml --prompt-id prose_en --max-new-bytes 128
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from byteprof import optim  # noqa: E402
from byteprof.config import load_config  # noqa: E402
from byteprof.optim import bolmo_loop  # noqa: E402
from byteprof.runner import generate_checked, load_model, load_prompts, set_seed  # noqa: E402


def snapshot_model(model):
    enc, dec = model.model.local_encoder, model.model.local_decoder
    states = [tuple(t.clone() for t in ls["xlstm"]["state"]) for ls in enc.layer_states + dec.layer_states]
    return {"states": states, "last_value": dec.last_value.clone()}


def snapshot_loop(loop):
    return {"states": [tuple(t.clone() for t in st) for st in loop.enc_states + loop.dec_states],
            "last_value": loop.last_value.clone()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--prompt-id", default="prose_en")
    ap.add_argument("--max-new-bytes", type=int, default=128)
    args = ap.parse_args()
    cfg = load_config(args.config, {"max_new_bytes": args.max_new_bytes})
    prompt = next(p for p in load_prompts(cfg.prompts_file) if p["id"] == args.prompt_id)
    model, tok = load_model(cfg)
    sync = torch.cuda.synchronize

    ref_snaps, ref_sub = [], []

    def hook(module, a, kwargs, output):
        ref_snaps.append(snapshot_model(model))
        e = kwargs.get("expanded_input_ids")
        ref_sub.append(e.flatten().tolist() if e is not None else None)

    h = model.model.register_forward_hook(hook, with_kwargs=True)
    set_seed(cfg.seed)
    ref = generate_checked(model, tok, prompt["text"], cfg, sync, want_text=True)
    h.remove()
    print(f"reference: {ref.new_bytes} new bytes, {len(ref_snaps)} model steps")

    handle = optim.apply(model, tok, cfg)
    mine_snaps, mine_sub = [], []

    def debug(i, info):
        mine_snaps.append(snapshot_loop(info["loop"]))
        mine_sub.append(info["sub"])

    # re-create the generate() closure with the debug callback
    model.generate = bolmo_loop.make_generate(model, tok, handle.runner, handle.loop, handle.glob, handle.pieces,
                                              handle.automaton, debug=debug)
    set_seed(cfg.seed)
    mine = generate_checked(model, tok, prompt["text"], cfg, sync, want_text=True)
    print(f"minimal:   {mine.new_bytes} new bytes, {len(mine_snaps)} decode steps")

    bad = 0
    for i, snap in enumerate(mine_snaps):
        k = i + 1  # reference step 0 is the prefill forward
        r = ref_snaps[k]
        problems = []
        if ref_sub[k] != [mine_sub[i]]:
            problems.append(f"subword id ref={ref_sub[k]} mine={mine_sub[i]}")
        for j, (a, b) in enumerate(zip(r["states"], snap["states"])):
            for n, (x, y) in enumerate(zip(a, b)):
                if not torch.equal(x, y):
                    problems.append(f"state layer {j} tensor {n} max|diff|={(x.float() - y.float()).abs().max().item():.3g}")
        if not torch.equal(r["last_value"], snap["last_value"]):
            problems.append("last_value differs")
        if problems:
            bad += 1
            if bad <= 3:
                print(f"byte {i}: " + "; ".join(problems))
    print("text identical:", ref.text == mine.text, "| steps with state mismatch:", bad, "of", len(mine_snaps))
    return 0 if bad == 0 and ref.text == mine.text else 1


if __name__ == "__main__":
    sys.exit(main())
