"""Stage 2 helper: time of one big matvec phase (bf16, K = 2048) vs its size, per-phase kernel in a CUDA graph.

Fits t = a + bytes / BW to separate the fixed cost of one phase (launch, ramp, tail) from the asymptotic bandwidth of the
streaming loop. Output: results/megakernel/asym.json.   python -X utf8 tools/mk_asym.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from byteprof.bench import cuda_rt, stream_bench as sbn  # noqa: E402


def main() -> None:
    st = sbn.Stack("bf16")
    f = sbn.module().func("phase_bf16", sbn.SMEM_EXTRA)
    bw = sbn.copy_bandwidth()
    rows_list = (1024, 2048, 4096, 6144, 12288, 24576, 49152)
    res = {"copy_bandwidth": bw, "points": []}
    for rows in rows_list:
        w = torch.randn(rows, 2048, device="cuda").bfloat16()
        out = torch.zeros(rows, device="cuda")
        ph = st._matvec([sbn.QMat(w.view(torch.uint8).reshape(rows, 4096), None, None, rows, 2048)], [out], 2048, st.buf["x"])
        dev = torch.frombuffer(bytearray((sbn.CPhase * 1)(ph)), dtype=torch.uint8).cuda()
        ts = []
        for _ in range(5):
            def fn():
                for _ in range(10):
                    cuda_rt.launch(f, 38, 256, [dev, 0, 0, 0], smem=sbn.SMEM_EXTRA)
            g = st.graph_of(fn)
            ts.append(sbn.time_graph(g, reps=10, trials=5) / 10)
        t = sorted(ts)[2]
        by = rows * 4096
        res["points"].append({"rows": rows, "bytes": by, "us": t, "GBs": by / t / 1e3})
        print(f"rows {rows:6d} {by / 1e6:7.1f} MB {t:8.1f} us {by / t / 1e3:6.1f} GB/s", flush=True)
    xs = [p["bytes"] for p in res["points"] if p["bytes"] > 20e6]
    ys = [p["us"] for p in res["points"] if p["bytes"] > 20e6]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    res["fit_fixed_us"] = my - slope * mx
    res["fit_GBs"] = 1e-3 / slope
    print(f"fit (>20 MB): fixed {res['fit_fixed_us']:.2f} us per phase, asymptote {res['fit_GBs']:.1f} GB/s")
    Path(ROOT / "results/megakernel/asym.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
