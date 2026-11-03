"""Launches tools/nsys_trace.py under Nsight Systems (Windows). Usage:

  python -X utf8 tools/run_nsys.py --config configs/fair/bolmo-all-int4.toml --out results/fair/nsys/all-int4

Passes the exact `ptxas --version` output to the target (see _patch_triton_ptxas in nsys_trace.py).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NSYS = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2023.4.4\target-windows-x64\nsys.exe")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generations", default="2")
    ap.add_argument("--bytes", default="128")
    ap.add_argument("--graph-trace", default="node")
    args = ap.parse_args()
    ptxas = ROOT / ".venv/Lib/site-packages/triton/backends/nvidia/bin/ptxas.exe"
    (ROOT / "results/fair/nsys").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/fair/nsys/ptxas_version.txt").write_bytes(subprocess.check_output([str(ptxas), "--version"]))
    env = dict(os.environ)
    cmd = [str(NSYS), "profile", "--trace=cuda,nvtx", f"--cuda-graph-trace={args.graph_trace}",
           "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-o", args.out, "--force-overwrite", "true",
           sys.executable, "-X", "utf8", str(ROOT / "tools" / "nsys_trace.py"), "--config", args.config,
           "--generations", args.generations, "--bytes", args.bytes]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
