"""Stage C3 driver: runs the Bolmo minimal loop with opt-in NVTX ranges, for Nsight Systems.

Ranges are added by wrapping from the outside (nothing in byteprof/optim changes, so the speed runs
carry no instrumentation): "byte" (one decoded byte: from the start of its enc_m replay to the end of
its sync), "replay:<graph>" around every graph replay, "sync" around Event.synchronize.
Only the traced generations are inside cudaProfilerStart/Stop.

  nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi \
      --capture-range-end=stop -o results/fair/nsys/all-int4 --force-overwrite true \
      python -X utf8 tools/nsys_trace.py --config configs/fair/bolmo-all-int4.toml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from byteprof import optim, runner  # noqa: E402
from byteprof.config import load_config  # noqa: E402

push, pop = torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop


def _patch_triton_ptxas() -> None:
    """Under nsys on this machine the child process `ptxas.exe --version` returns nothing, so Triton raises
    "Cannot find ptxas.exe" before every kernel lookup. tools/run_nsys.py writes the real output (taken outside
    nsys) to ptxas_version.txt; this returns it for exactly that call, so Triton's cache keys, and therefore
    the already compiled kernels, are unchanged. Every other subprocess call is untouched."""
    import subprocess

    f = Path("results/fair/nsys/ptxas_version.txt")
    if not f.is_file():
        return
    text = f.read_bytes()
    orig = subprocess.check_output

    def check_output(cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and len(cmd) == 2 and str(cmd[0]).lower().endswith("ptxas.exe") \
                and cmd[1] == "--version":
            return text
        return orig(cmd, *a, **k)

    subprocess.check_output = check_output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--bytes", type=int, default=128)
    args = ap.parse_args()
    _patch_triton_ptxas()
    cfg = replace(load_config(Path(args.config)), max_new_bytes=args.bytes)
    model, tok = runner.load_model(cfg)
    handle = optim.apply(model, tok, cfg)
    rn = handle.runner
    sync = torch.cuda.synchronize
    prompts = runner.load_prompts(cfg.prompts_file)
    for p in prompts:  # warmup, no tracing
        runner.generate_checked(model, tok, p["text"], cfg, sync)

    state = {"byte_open": False}
    orig_run = rn.run

    def run(name: str) -> None:
        if name == "enc_m":
            if state["byte_open"]:
                pop()
            push("byte")
            state["byte_open"] = True
        push(f"replay:{name}")
        orig_run(name)
        pop()

    rn.run = run
    orig_sync = torch.cuda.Event.synchronize

    def synchronize(self, *a, **k):
        push("sync")
        try:
            return orig_sync(self, *a, **k)
        finally:
            pop()

    torch.cuda.Event.synchronize = synchronize
    torch.cuda.profiler.start()
    for i in range(args.generations):
        push(f"generation{i}")
        runner.generate_checked(model, tok, prompts[0]["text"], cfg, sync)
        if state["byte_open"]:
            pop()
            state["byte_open"] = False
        pop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    torch.cuda.Event.synchronize = orig_sync
    print("traced", args.generations, "generations of", args.bytes, "bytes")


if __name__ == "__main__":
    import traceback

    try:
        main()
    except BaseException:  # nsys swallows the target's output; keep the error somewhere visible
        Path("results/fair/nsys/trace_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
