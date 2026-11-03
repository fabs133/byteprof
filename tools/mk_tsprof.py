"""Stage 2 helper: per-phase timeline of the persistent kernel (globaltimer stamps per block and phase).

For each phase: spread of the start times, compute time (mean / max over blocks), wait time at the barrier. Medians over 20 launches.
Output: results/megakernel/tsprof-<q>.json.   python -X utf8 tools/mk_tsprof.py --q bf16 [--prefetch 0] [--sw-kib 80]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from byteprof.bench import stream_bench as sbn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="bf16")
    ap.add_argument("--prefetch", type=int, default=0)
    ap.add_argument("--sw-kib", type=int, default=80)
    a = ap.parse_args()
    st = sbn.Stack(a.q)
    st.sw_bytes = a.sw_kib * 1024
    G = torch.cuda.get_device_properties(0).multi_processor_count
    ts = torch.zeros(st.nph * G * 3, dtype=torch.int64, device="cuda")
    for _ in range(30):
        st.mega(a.prefetch)
    torch.cuda.synchronize()
    runs = []
    for _ in range(20):
        st.mega(a.prefetch, ts=ts)
        torch.cuda.synchronize()
        runs.append(ts.cpu().numpy().reshape(st.nph, G, 3).astype(np.float64))
    phases = []
    for p in range(st.nph):
        s = np.median([r[p, :, 0] for r in runs], axis=0)
        e = np.median([r[p, :, 1] for r in runs], axis=0)
        w = np.median([r[p, :, 2] for r in runs], axis=0) if p < st.nph - 1 else e
        kind = int(st.phases[p].kind)
        phases.append({"phase": p, "kind": kind, "start_skew_us": float(s.max() - s.min()) / 1e3,
                       "compute_mean_us": float((e - s).mean()) / 1e3, "compute_max_us": float((e - s).max()) / 1e3,
                       "wait_mean_us": float((w - e).mean()) / 1e3, "wait_min_us": float((w - e).min()) / 1e3})
    mlstm = [p["compute_max_us"] for p in phases if p["kind"] == 1]
    out = {"q": a.q, "prefetch": a.prefetch, "sw_kib": a.sw_kib, "phases": phases,
           "mlstm_phase_us": float(np.median(mlstm)),
           "matvec_tail_us": float(np.median([p["compute_max_us"] - p["compute_mean_us"] for p in phases if p["kind"] == 0])),
           "barrier_wait_min_us": float(np.median([p["wait_min_us"] for p in phases[:-1]]))}
    Path(ROOT / f"results/megakernel/tsprof-{a.q}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print({k: v for k, v in out.items() if k != "phases"})


if __name__ == "__main__":
    main()
