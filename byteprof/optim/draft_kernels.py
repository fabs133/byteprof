"""Triton kernel for the drafting loop (imported lazily by draft_core on CUDA only)."""

import triton
import triton.language as tl


@triton.jit
def commit_kernel(c_hist, c_cur, n_hist, n_cur, m_hist, m_cur, l_ptr, C_N, N_N, M_N, BLOCK: tl.constexpr):
    """cur <- hist[slot] for one layer's state (c, n, m); slot is read from the device (the accepted length L)."""
    pid = tl.program_id(0)
    slot = tl.load(l_ptr).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    cm = offs < C_N
    tl.store(c_cur + offs, tl.load(c_hist + slot * C_N + offs, mask=cm), mask=cm)
    nm = offs < N_N
    tl.store(n_cur + offs, tl.load(n_hist + slot * N_N + offs, mask=nm), mask=nm)
    mm = offs < M_N
    tl.store(m_cur + offs, tl.load(m_hist + slot * M_N + offs, mask=mm), mask=mm)
