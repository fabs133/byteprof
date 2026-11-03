"""Stage A0: GPU time of a graphed drafting round (A_K + C_K) per kernel class, from tools/nsys_rounds.py traces.

Classes (by kernel name; the lm_head is the last linear kernel of a round, the only one that is not a layer linear):
  mlstm_step   recurrent_step_fw_kernel / multistep_fw_kernel (the state update; the steps path includes the history write of the state after each position)
  commit       commit_kernel (history slot L -> live state) / lazy_commit_kernel (recompute over 0..L)
  linear       gemm / tinygemm / Kernel2 / gemv / splitKreduce kernels (bf16 matmul, int4 tinygemm), layer linears
  lm_head      the last linear kernel of the round (+ its splitKreduce)
  elementwise  vectorized_/unrolled_/elementwise_kernel, reduce_kernel (norms, gates, residuals, embeddings, argmax)
  scan         DeviceScan* (acceptance cumprod) / accept_kernel (fused acceptance)
  other        everything else (index_select, ...)
Usage: python -I tools/nsys_rounds_analyze.py results/parallel/nsys/bf16.sqlite --out results/parallel/nsys/bf16-classes.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3

LINEAR = ("gemm", "Kernel2", "gemv", "splitK", "tinygemm")
CLASSES = ["mlstm_step", "commit", "linear", "lm_head", "elementwise", "scan", "other"]


def classify(name: str) -> str:
    if name.startswith(("recurrent_step_fw_kernel", "multistep_fw_kernel")):
        return "mlstm_step"
    if name.startswith(("commit_kernel", "lazy_commit_kernel")):
        return "commit"
    if any(t in name for t in LINEAR):
        return "linear"
    if "elementwise" in name or name.startswith("reduce_kernel"):
        return "elementwise"
    if "DeviceScan" in name or name.startswith("accept_kernel"):
        return "scan"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--out")
    a = ap.parse_args()
    c = sqlite3.connect(a.sqlite)
    sid = dict(c.execute("select id, value from StringIds"))
    rng = c.execute("select start, end, text from NVTX_EVENTS where text like 'block:%' order by start").fetchall()
    kern = c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    res: dict = {}
    for s, e, text in rng:
        _, k_, reps_ = text.split(":")
        k, reps = int(k_), int(reps_)
        ks = [(ks_, ke, sid.get(sn, str(sn))) for ks_, ke, sn in kern if s <= ks_ <= e]
        assert len(ks) % reps == 0, (k, len(ks), reps)
        n = len(ks) // reps
        one = ks[:n]  # the block is periodic: classify the first round, the lm_head is its last linear kernel
        cls = [classify(nm) for _, _, nm in one]
        last = max(i for i, x in enumerate(cls) if x == "linear")
        while "splitK" in one[last][2]:
            last -= 1
        lm = {last} | ({last + 1} if "splitK" in one[last + 1][2] else set())
        row = {c_: [0, 0.0] for c_ in CLASSES}
        for i, (ks_, ke, nm) in enumerate(ks):
            cl = "lm_head" if (i % n) in lm else cls[i % n]
            row[cl][0] += 1
            row[cl][1] += (ke - ks_) / 1000
        span = (ks[-1][1] - ks[0][0]) / 1000
        res[k] = {"reps": reps, "kernels": n, "span_us": span / reps,
                  "kernel_sum_us": sum(v[1] for v in row.values()) / reps,
                  "classes": {c_: {"count": row[c_][0] / reps, "us": row[c_][1] / reps} for c_ in CLASSES}}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    ks_ = sorted(res)
    print("| class | " + " | ".join(f"K={k} us (n)" for k in ks_) + " |")
    print("|---|" + "---:|" * len(ks_))
    for c_ in CLASSES:
        print(f"| {c_} | " + " | ".join(f"{res[k]['classes'][c_]['us']:.1f} ({res[k]['classes'][c_]['count']:.0f})" for k in ks_) + " |")
    print("| kernel sum | " + " | ".join(f"{res[k]['kernel_sum_us']:.1f} ({res[k]['kernels']:.0f})" for k in ks_) + " |")
    print("| first-to-last span | " + " | ".join(f"{res[k]['span_us']:.1f}" for k in ks_) + " |")


if __name__ == "__main__":
    main()
