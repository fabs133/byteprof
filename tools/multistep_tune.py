"""Stage A1 helper: time of the fused multi-step kernel (and the lazy commit) for one layer at T positions, for several
(value block BV, num_warps) choices, inside a CUDA graph (back-to-back replays). Bolmo local layer geometry:
NH = 16, DQ = 64, DV = 128, bf16 inputs, fp32 state.

  python tools/multistep_tune.py
"""

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from byteprof.optim import draft_fused as fused  # noqa: E402
from byteprof.optim.mlstm_multistep import lazy_commit_kernel, multistep_fw_kernel  # noqa: E402

NH, DQ, DV = 16, 64, 128


def timed(fn, reps=200, trials=5) -> float:
    g = torch.cuda.CUDAGraph()
    fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    out = []
    for _ in range(trials):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / reps / 10 * 1000)
    return statistics.median(out)


def main() -> None:
    from mlstm_kernels.torch.utils import torch2triton_dtype

    dt = torch.bfloat16
    for t in (1, 2, 9, 33):
        q, k, v = (torch.randn(t, NH, DQ, device="cuda", dtype=dt), torch.randn(t, NH, DQ, device="cuda", dtype=dt),
                   torch.randn(t, NH, DV, device="cuda", dtype=dt))
        i, f = torch.randn(t, NH, 1, device="cuda", dtype=dt), torch.randn(t, NH, 1, device="cuda", dtype=dt)
        cur = (torch.randn(1, NH, DQ, DV, device="cuda") * 0.1, torch.randn(1, NH, DQ, device="cuda") * 0.1,
               torch.randn(1, NH, 1, device="cuda"))
        h = torch.zeros(t, NH, DV, device="cuda", dtype=dt)
        ns, ms = torch.zeros(NH, DQ, device="cuda"), torch.zeros(NH, device="cuda")
        length = torch.full((), t - 1, dtype=torch.long, device="cuda")
        for bv in (128, 64, 32, 16):
            for nw in (1, 2, 4, 8):
                def one():
                    multistep_fw_kernel[(DV // bv, NH)](
                        cur[0], cur[1], cur[2], q, k, v, i, f, h, ns, ms, t, q.stride(0), v.stride(0), i.stride(0),
                        h.stride(0), qk_scale=DQ ** -0.5, NH=NH, DQ=DQ, DV=DV, BQ=64, BV=bv, EPS=1e-6,
                        DTYPE=torch2triton_dtype(dt), DTYPE_STATE=torch2triton_dtype(torch.float32), num_warps=nw,
                        num_stages=1)

                def commit():
                    lazy_commit_kernel[(DV // bv, NH)](
                        cur[0], cur[1], cur[2], ns, ms, k, v, i, f, length, k.stride(0), v.stride(0), i.stride(0),
                        NH=NH, DQ=DQ, DV=DV, BQ=64, BV=bv, DTYPE=torch2triton_dtype(dt),
                        DTYPE_STATE=torch2triton_dtype(torch.float32), num_warps=nw, num_stages=1)

                try:
                    print(f"T={t:2d} BV={bv:3d} warps={nw}: pass1 {timed(one):6.2f} us   commit(L=T-1) {timed(commit):6.2f} us")
                except Exception as e:  # noqa: BLE001
                    print(f"T={t} BV={bv} warps={nw}: failed {type(e).__name__}")


if __name__ == "__main__":
    main()
