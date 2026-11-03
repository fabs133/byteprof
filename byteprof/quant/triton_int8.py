"""Fused int8 weight-only matrix-vector product (one activation row), Triton.

torch 2.6 has no CUDA `_weight_int8pack_mm` and torchao 0.9's int8 weight-only path dequantizes
in a separate step (slower than bf16 here, see FINDINGS.md Stage 1). This kernel reads the int8
weights once and converts them to fp32 in registers: y[n] = scale[n] * sum_k x[k] * w[n, k].
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_int8(x_ptr, w_ptr, s_ptr, y_ptr, N, K, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BN + tl.arange(0, BN)
    rmask = rows < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        cols = k0 + tl.arange(0, BK)
        cmask = cols < K
        x = tl.load(x_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + rows[:, None] * K + cols[None, :], mask=rmask[:, None] & cmask[None, :], other=0)
        acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    s = tl.load(s_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    tl.store(y_ptr + rows, (acc * s).to(y_ptr.dtype.element_ty), mask=rmask)


# (BN, BK, num_warps): best of a sweep over BN in 1..8, BK in 256..1024, num_warps in 1..8, on the four
# local-layer shapes (2048 and 2816 wide: (2, 1024, 2); 1024 outputs: (2, 512, 2)); CUDA-graph timed.
DEFAULT_CFG = (2, 1024, 2)


def default_cfg(n_out: int) -> tuple[int, int, int]:
    return (2, 512, 2) if n_out <= 1024 else DEFAULT_CFG


def gemv_int8(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor, cfg: tuple[int, int, int] | None = None,
              out: torch.Tensor | None = None) -> torch.Tensor:
    """x: [1, K] bf16, w: [N, K] int8, scale: [N] bf16 -> [1, N] bf16."""
    N, K = w.shape
    bn, bk, nw = cfg or default_cfg(N)
    y = out if out is not None else torch.empty(1, N, dtype=x.dtype, device=x.device)
    _gemv_int8[(triton.cdiv(N, bn),)](x, w, scale, y, N, K, BN=bn, BK=bk, num_warps=nw)
    return y


def quantize_rowwise_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-row int8: w ~ q * scale[:, None]; scale in the weight's dtype."""
    # round the scale to the weight dtype first, so q is quantized against the scale that is stored
    scale = (w.float().abs().amax(dim=1) / 127.0).clamp_min(1e-12).to(w.dtype)
    q = torch.round(w.float() / scale.float()[:, None]).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), scale
