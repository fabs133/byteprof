"""Stage C1 of TASK_fair_ratio_floor.md: minimum GPU cost per CUDA-graph kernel node on this machine.

Captures a graph of N trivial kernels (in-place add on a 1-element tensor; dependent chain, i.e. the
same serialization as a real model) and reports microseconds per node on replay. Also a variant with
independent tensors (still serialized on one stream) to see whether the dependency matters.
Run: python -X utf8 tools/graph_node_floor.py [--out results/fair/node_floor.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPLAYS = 50


def per_node_us(n: int, chain: bool) -> float:
    xs = [torch.zeros(1, device="cuda")] if chain else [torch.zeros(1, device="cuda") for _ in range(n)]

    def body() -> None:
        for i in range(n):
            xs[0 if chain else i].add_(1.0)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(REPLAYS):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1000 / REPLAYS / n)
    return sorted(times)[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/fair/node_floor.json")
    args = ap.parse_args()
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": []}
    for n in (100, 300, 1000):
        for chain in (True, False):
            us = per_node_us(n, chain)
            res["rows"].append({"nodes": n, "dependent_chain": chain, "us_per_node": us})
            print(f"N={n:5d} {'chain' if chain else 'indep':5s}: {us:.2f} us per node", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
