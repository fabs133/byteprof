"""Python wrappers for the fused verification kernels (mlstm_multistep.py); CUDA only, imported lazily by
bolmo_draft when `draft.verify = "fused"`."""

from __future__ import annotations

import torch

# Tuned with tools/multistep_tune.py on the RTX 3060 Ti (T = 1..33 positions, Bolmo local layer geometry): narrow value
# blocks (more programs, less state per program) and a single warp for the first pass, two warps for the commit.
BV = 16
WARPS_FW, WARPS_COMMIT = 1, 2


def _geometry(q: torch.Tensor, v: torch.Tensor) -> tuple[int, int, int, int, int, int]:
    from mlstm_kernels.triton.kernel_param_heuristics import get_head_dim_block_size

    _, nh, dq = q.shape
    dv = v.shape[-1]
    bq = get_head_dim_block_size(head_dim=dq, min_block_size=64)
    bv = get_head_dim_block_size(head_dim=dv, min_block_size=64)
    if dq not in (bq, 2 * bq):
        raise NotImplementedError(f"the fused kernel needs head_dim_qk of one or two blocks ({dq} vs {bq})")
    del bv
    return nh, dq, dv, bq, BV, dv // BV


def multistep_fw(cur: tuple[torch.Tensor, ...], q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, i: torch.Tensor,
                 f: torch.Tensor, h: torch.Tensor, nstart: torch.Tensor, mstart: torch.Tensor, eps: float) -> None:
    """T = q.shape[0] positions from the state `cur` (not modified); h (>= T, NH, DV) receives the cell outputs;
    the start n and m are copied to nstart (NH, DQ) and mstart (NH,) for `lazy_commit`."""
    from mlstm_kernels.torch.utils import torch2triton_dtype

    from .mlstm_multistep import multistep_fw_kernel

    nh, dq, dv, bq, bv, nvb = _geometry(q, v)
    multistep_fw_kernel[(nvb, nh)](
        cur[0], cur[1], cur[2], q, k, v, i, f, h, nstart, mstart, q.shape[0],
        q.stride(0), v.stride(0), i.stride(0), h.stride(0), qk_scale=dq ** -0.5,
        NH=nh, DQ=dq, DV=dv, BQ=bq, BV=bv, EPS=eps, DTYPE=torch2triton_dtype(q.dtype),
        DTYPE_STATE=torch2triton_dtype(cur[0].dtype), num_warps=WARPS_FW, num_stages=1)


def lazy_commit(cur: tuple[torch.Tensor, ...], k: torch.Tensor, v: torch.Tensor, i: torch.Tensor, f: torch.Tensor,
                length: torch.Tensor, nstart: torch.Tensor, mstart: torch.Tensor) -> None:
    """cur <- the state after positions 0 .. L (L = `length`, a 0-d device integer), recomputed from the start state."""
    from mlstm_kernels.torch.utils import torch2triton_dtype

    from .mlstm_multistep import lazy_commit_kernel

    nh, dq, dv, bq, bv, nvb = _geometry(k, v)
    lazy_commit_kernel[(nvb, nh)](
        cur[0], cur[1], cur[2], nstart, mstart, k, v, i, f, length,
        k.stride(0), v.stride(0), i.stride(0),
        NH=nh, DQ=dq, DV=dv, BQ=bq, BV=bv, DTYPE=torch2triton_dtype(k.dtype),
        DTYPE_STATE=torch2triton_dtype(cur[0].dtype), num_warps=WARPS_COMMIT, num_stages=1)


def accept(ids: torch.Tensor, tok: torch.Tensor, out: torch.Tensor, s: int, smax: int) -> None:
    """out[:s] <- tok[:s], out[smax] <- accepted length (see accept_kernel)."""
    from .mlstm_multistep import accept_kernel

    accept_kernel[(1,)](ids, tok, out, S=s, SMAX=smax, num_warps=1)
