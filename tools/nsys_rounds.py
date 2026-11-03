"""Stage A0 driver: replays the graphed drafting round pieces (A_K = encd{K}, C_K = decd{K}n; K = 0: enc_m, dec_n)
under NVTX ranges "round:<K>" for Nsight Systems. Each "block:<K>:<reps>" range is <reps> back-to-back A_K + C_K replays (after 300 untraced
warm-up rounds) closed by a synchronize, so every kernel of the block lies inside it.

  python -X utf8 tools/nsys_rounds.py --config configs/draft/bolmo-bf16-combined-k8.toml --out results/parallel/nsys/bf16
(launches itself under nsys when --out is given; the child is started with --child)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
NSYS = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.4.4\target-windows-x64\nsys.exe")


def child(a: argparse.Namespace) -> None:
    import torch

    from byteprof import optim, runner
    from byteprof.config import load_config
    sys.path.insert(0, str(ROOT / "tools"))
    from nsys_trace import _patch_triton_ptxas

    _patch_triton_ptxas()
    cfg = load_config(ROOT / a.config, {"max_new_bytes": 96})
    model, tok = runner.load_model(cfg)
    handle = optim.apply(model, tok, cfg)
    rn, engine = handle.runner, handle.drafting["engine"]
    prompt = runner.load_prompts(ROOT / "prompts/default.jsonl")[0]["text"]
    runner.generate_checked(model, tok, prompt, cfg, torch.cuda.synchronize)
    g = rn.graphs
    ks = [int(k) for k in a.ks.split(",")]
    push, pop = torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop
    with torch.inference_mode():
        for k in ks:  # warm
            g["enc_m" if k == 0 else f"encd{k}"].replay()
            g["dec_n" if k == 0 else f"decd{k}n"].replay()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        for k in ks:
            ea = g["enc_m" if k == 0 else f"encd{k}"]
            ec = g["dec_n" if k == 0 else f"decd{k}n"]
            for _ in range(300):  # hot clocks before the traced block
                ea.replay()
                ec.replay()
            torch.cuda.synchronize()
            push(f"block:{k}:{a.reps}")
            for _ in range(a.reps):
                ea.replay()
                ec.replay()
            torch.cuda.synchronize()
            pop()
        torch.cuda.profiler.stop()
    print("traced", ks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out")
    ap.add_argument("--ks", default="0,1,4,8")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.child:
        try:
            child(a)
        except BaseException:
            import traceback

            Path("results/parallel/nsys").mkdir(parents=True, exist_ok=True)
            Path("results/parallel/nsys/trace_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        return 0
    ptxas = ROOT / ".venv/Lib/site-packages/triton/backends/nvidia/bin/ptxas.exe"
    (ROOT / "results/fair/nsys").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/fair/nsys/ptxas_version.txt").write_bytes(subprocess.check_output([str(ptxas), "--version"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(NSYS), "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
           "--capture-range-end=stop", "-o", a.out, "--force-overwrite", "true", sys.executable, "-X", "utf8",
           str(Path(__file__)), "--child", "--config", a.config, "--ks", a.ks, "--reps", str(a.reps)]
    r = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ), capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
