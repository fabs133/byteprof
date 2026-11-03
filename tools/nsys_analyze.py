"""Stage C2/C3: per-decoded-byte decomposition from an nsys sqlite export (tools/run_nsys.py trace).

Per NVTX "byte" window (from one enc_m replay launch to the next; it contains the host work, the graph
launches, the GPU work and the sync) it reports, averaged over bytes:
  window, kernels and memcpys per byte, GPU kernel time, time in kernels < 10 us, GPU busy time,
  gaps between GPU operations inside the active span, idle before the first GPU operation (host launch
  latency) and after the last one (sync return + host loop), NVTX sync time, D2H copy time;
and kernels per graph (enc_m / dec / glob), from the correlation id of each cudaGraphLaunch.
Usage: python -X utf8 tools/nsys_analyze.py results/fair/nsys/all-int4.sqlite [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict

SHORT_US = 10.0


def union_length(intervals: list[tuple[int, int]]) -> int:
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--json")
    args = ap.parse_args()
    c = sqlite3.connect(args.sqlite)
    strings = dict(c.execute("select id, value from StringIds"))
    nv = c.execute("select start, end, text from NVTX_EVENTS where end is not null order by start").fetchall()
    bytes_ = [(s, e) for s, e, t in nv if t == "byte"]
    syncs = [(s, e) for s, e, t in nv if t == "sync"]
    replays = [(s, e, t.split(":")[1]) for s, e, t in nv if t.startswith("replay:")]
    kernels = c.execute("select start, end, shortName, correlationId, demangledName from CUPTI_ACTIVITY_KIND_KERNEL "
                        "order by start").fetchall()
    copies = c.execute("select start, end, copyKind, bytes, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY "
                       "order by start").fetchall()
    rt_cols = [r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_RUNTIME)")]
    rt = c.execute("select start, end, nameId, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
    launch_ids = {i for i, v in strings.items() if v and v.startswith("cudaGraphLaunch")}
    # correlation id of each graph launch -> label of the NVTX replay range that contains the launch call
    label_of_corr: dict[int, str] = {}
    ri = 0
    for s, e, name_id, corr in rt:
        if name_id in launch_ids:
            for rs, re_, lab in replays:
                if rs <= s <= re_:
                    label_of_corr[corr] = lab
                    break
    ks = [k[0] for k in kernels]
    import bisect

    rows = []
    per_graph: dict[str, list[int]] = defaultdict(list)
    per_graph_t: dict[str, list[float]] = defaultdict(list)
    gem_t, gem_n, shrt_n, shrt_t = (defaultdict(list) for _ in range(4))
    names_n: Counter = Counter()
    names_t: Counter = Counter()
    n_bytes = 0
    for (ws, we) in bytes_[1:-1]:  # drop the first and last window (edge effects)
        lo, hi = bisect.bisect_left(ks, ws), bisect.bisect_left(ks, we)
        kk = [k for k in kernels[lo:hi]]
        cc = [x for x in copies if ws <= x[0] < we]
        if not kk:
            continue
        gpu_ops = [(k[0], k[1]) for k in kk] + [(x[0], x[1]) for x in cc]
        first = min(s for s, _ in gpu_ops)
        last = max(e for _, e in gpu_ops)
        span = last - first
        busy = union_length(gpu_ops)
        ktime = sum(k[1] - k[0] for k in kk)
        short = [k for k in kk if (k[1] - k[0]) < SHORT_US * 1000]
        d2d = [x for x in cc if x[2] == 8]
        d2h = [x for x in cc if x[2] == 2]
        sy = [e - s for s, e in syncs if ws <= s < we]
        rows.append({
            "window_us": (we - ws) / 1000, "kernels": len(kk), "copies": len(cc), "d2d_copies": len(d2d),
            "kernel_us": ktime / 1000, "short_kernels": len(short), "short_us": sum(k[1] - k[0] for k in short) / 1000,
            "busy_us": busy / 1000, "span_us": span / 1000, "gaps_in_span_us": (span - busy) / 1000,
            "pre_idle_us": (first - ws) / 1000, "post_idle_us": (we - last) / 1000,
            "sync_us": sum(sy) / 1000, "d2h_us": sum(x[1] - x[0] for x in d2h) / 1000,
            "d2d_us": sum(x[1] - x[0] for x in d2d) / 1000,
        })
        by_label: dict[str, list] = defaultdict(list)
        for k in kk:
            by_label[label_of_corr.get(k[3], "?")].append(k)
        for lab, lst in by_label.items():
            per_graph[lab].append(len(lst))
            per_graph_t[lab].append(sum(k[1] - k[0] for k in lst) / 1000)
            gemm = [k for k in lst if "tinygemm" in strings.get(k[2], "") or "_gemv_int8" in strings.get(k[2], "")]
            gem_t[lab].append(sum(k[1] - k[0] for k in gemm) / 1000)
            gem_n[lab].append(len(gemm))
            shrt = [k for k in lst if (k[1] - k[0]) < SHORT_US * 1000]
            shrt_n[lab].append(len(shrt))
            shrt_t[lab].append(sum(k[1] - k[0] for k in shrt) / 1000)
        for k in kk:
            n = strings.get(k[2], str(k[2]))
            names_n[n] += 1
            names_t[n] += k[1] - k[0]
        n_bytes += 1
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    print(f"windows analysed: {len(rows)} (of {len(bytes_)} bytes); graph launches labelled: {len(label_of_corr)}")
    for k, v in mean.items():
        print(f"  {k:18s} {v:10.1f}")
    print("kernels per graph launch (mean count / mean GPU us), per byte it is weighted by launches per byte:")
    out_graph = {}
    for lab in sorted(per_graph):
        cnt = [x for x in per_graph[lab]]
        # launches of this graph per window
        print(f"  {lab:8s} windows with it: {len(cnt):4d}  mean kernels per window {sum(cnt) / len(cnt):8.1f}  "
              f"mean kernel us per window {sum(per_graph_t[lab]) / len(cnt):9.1f}")
        m = lambda v: sum(v) / len(v)  # noqa: E731
        out_graph[lab] = {"windows": len(cnt), "kernels": m(cnt), "kernel_us": m(per_graph_t[lab]),
                          "gemm_kernels": m(gem_n[lab]), "gemm_us": m(gem_t[lab]),
                          "short_kernels": m(shrt_n[lab]), "short_us": m(shrt_t[lab])}
        print(f"           gemm kernels {m(gem_n[lab]):6.1f} ({m(gem_t[lab]):7.1f} us), short (<10us) kernels "
              f"{m(shrt_n[lab]):6.1f} ({m(shrt_t[lab]):7.1f} us)")
    total = sum(names_t.values())
    print("top kernels by GPU time (share of kernel time, calls per byte, mean us):")
    top = []
    for n, t in names_t.most_common(12):
        print(f"  {t / total * 100:5.1f}%  {names_n[n] / n_bytes:7.2f}/byte  {t / names_n[n] / 1000:7.2f} us  {n[:90]}")
        top.append({"name": n, "share": t / total, "calls_per_byte": names_n[n] / n_bytes, "mean_us": t / names_n[n] / 1000})
    hist = Counter()
    for n in names_n:
        pass
    if args.json:
        json.dump({"per_byte_mean": mean, "per_graph": out_graph, "top_kernels": top, "n_windows": len(rows)},
                  open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
