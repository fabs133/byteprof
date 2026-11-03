"""Fused multi-step mLSTM recurrence for the drafting loop (TASK_parallel_verify.md, Stage A1). CUDA only; imported lazily.

`multistep_fw_kernel`: one launch per layer loops over T positions with the (c, n, m) state held in registers and
writes only the cell outputs h_t. It uses the formulation, stabilization, dtypes and block structure of the library's
`recurrent_step_fw_kernel` (mlstm_kernels 2.0.6, fused step): per position the gate math in fp32 then cast to the
input dtype, the state update in the state dtype, the DHQK dimension in two blocks of `BQ` whose partial sums
are added in the same order. It never writes the state.

`lazy_commit_kernel`: recomputes the state from the start state over positions 0 .. L (L read from device memory, so
the trip count is dynamic and the launch fits in a CUDA graph) and writes it into the live state. The start n and m
were saved by the first pass (`nstart`, `mstart`): the live n and m are overwritten by the programs of the first
value block while the programs of the second value block still need them. The live c tile of a program is read before
it is written, and tiles do not overlap, so c is updated in place.
"""

import triton
import triton.language as tl


@triton.jit
def _gates(i_ptr, f_ptr, t, str_t_if, head, m_old, DTYPE: tl.constexpr):
    off = t * str_t_if + head
    f_val = tl.load(f_ptr + off).to(tl.float32)
    i_val = tl.load(i_ptr + off).to(tl.float32)
    f_log = tl.log(tl.sigmoid(f_val))
    m_new = tl.maximum(f_log + m_old, i_val)
    max_val = tl.exp(-m_new.to(tl.float32)).to(DTYPE)
    f_act = tl.exp(f_log + m_old - m_new).to(DTYPE)
    i_act = tl.exp(i_val - m_new).to(DTYPE)
    return m_new, max_val, f_act, i_act


@triton.jit
def multistep_fw_kernel(
    c_ptr, n_ptr, m_ptr,  # start state (1, NH, DQ, DV), (1, NH, DQ), (1, NH, 1); read only
    q_ptr, k_ptr, v_ptr, i_ptr, f_ptr,  # (T, NH, DQ), (T, NH, DQ), (T, NH, DV), (T, NH, 1), (T, NH, 1)
    h_ptr,  # (T, NH, DV) out
    nstart_ptr, mstart_ptr,  # (NH, DQ), (NH,) fp32: copy of the start n and m for the lazy commit
    T,
    str_t_qk, str_t_v, str_t_if, str_t_h,
    qk_scale: tl.constexpr,
    NH: tl.constexpr, DQ: tl.constexpr, DV: tl.constexpr, BQ: tl.constexpr, BV: tl.constexpr,
    EPS: tl.constexpr, DTYPE: tl.constexpr, DTYPE_STATE: tl.constexpr,
):
    TWO: tl.constexpr = DQ > BQ  # the DHQK dimension is one block (Bolmo: 64) or two (the unit tests also use 128)
    idx_v, head = tl.program_id(0), tl.program_id(1)
    offs_q = tl.arange(0, BQ)
    offs_v = idx_v * BV + tl.arange(0, BV)
    c_base = c_ptr + head * DQ * DV
    c0 = tl.load(c_base + offs_q[:, None] * DV + offs_v[None, :]).to(DTYPE_STATE)
    n0 = tl.load(n_ptr + head * DQ + offs_q).to(DTYPE_STATE)
    if TWO:
        c1 = tl.load(c_base + (BQ + offs_q[:, None]) * DV + offs_v[None, :]).to(DTYPE_STATE)
        n1 = tl.load(n_ptr + head * DQ + BQ + offs_q).to(DTYPE_STATE)
    else:
        c1 = c0
        n1 = n0
    m_old = tl.load(m_ptr + head)
    if idx_v == 0:
        tl.store(nstart_ptr + head * DQ + offs_q, n0)
        if TWO:
            tl.store(nstart_ptr + head * DQ + BQ + offs_q, n1)
        tl.store(mstart_ptr + head, m_old)
    for t in range(0, T):
        m_new, max_val, f_act, i_act = _gates(i_ptr, f_ptr, t, str_t_if, head, m_old, DTYPE)
        v_val = tl.load(v_ptr + t * str_t_v + head * DV + offs_v)
        qk_off = t * str_t_qk + head * DQ
        k0 = tl.load(k_ptr + qk_off + offs_q)
        c0 = f_act * c0 + i_act * (k0[:, None] * v_val[None, :])
        n0 = f_act * n0 + i_act * k0
        if TWO:
            k1 = tl.load(k_ptr + qk_off + BQ + offs_q)
            c1 = f_act * c1 + i_act * (k1[:, None] * v_val[None, :])
            n1 = f_act * n1 + i_act * k1
        h_num = tl.zeros((BV,), dtype=tl.float32)
        qn = tl.zeros((1,), dtype=tl.float32)
        q0 = tl.load(q_ptr + qk_off + offs_q) * qk_scale
        h_num += tl.sum(q0[:, None] * c0.to(dtype=DTYPE), axis=0)
        qn += tl.sum(q0 * n0.to(dtype=DTYPE))
        if TWO:
            q1 = tl.load(q_ptr + qk_off + BQ + offs_q) * qk_scale
            h_num += tl.sum(q1[:, None] * c1.to(dtype=DTYPE), axis=0)
            qn += tl.sum(q1 * n1.to(dtype=DTYPE))
        h = tl.fdiv(h_num, tl.maximum(tl.abs(qn), max_val) + EPS)
        tl.store(h_ptr + t * str_t_h + head * DV + offs_v, h.to(DTYPE))
        m_old = m_new


@triton.jit
def lazy_commit_kernel(
    c_ptr, n_ptr, m_ptr,  # live state: c is read and written in place; n and m are written
    nstart_ptr, mstart_ptr,
    k_ptr, v_ptr, i_ptr, f_ptr,
    l_ptr,  # device int64: accepted length L; positions 0 .. L are applied
    str_t_qk, str_t_v, str_t_if,
    NH: tl.constexpr, DQ: tl.constexpr, DV: tl.constexpr, BQ: tl.constexpr, BV: tl.constexpr,
    DTYPE: tl.constexpr, DTYPE_STATE: tl.constexpr,
):
    TWO: tl.constexpr = DQ > BQ
    idx_v, head = tl.program_id(0), tl.program_id(1)
    offs_q = tl.arange(0, BQ)
    offs_v = idx_v * BV + tl.arange(0, BV)
    c_base = c_ptr + head * DQ * DV
    c0 = tl.load(c_base + offs_q[:, None] * DV + offs_v[None, :]).to(DTYPE_STATE)
    n0 = tl.load(nstart_ptr + head * DQ + offs_q).to(DTYPE_STATE)
    if TWO:
        c1 = tl.load(c_base + (BQ + offs_q[:, None]) * DV + offs_v[None, :]).to(DTYPE_STATE)
        n1 = tl.load(nstart_ptr + head * DQ + BQ + offs_q).to(DTYPE_STATE)
    else:
        c1 = c0
        n1 = n0
    m_old = tl.load(mstart_ptr + head)
    n_pos = tl.load(l_ptr).to(tl.int32) + 1
    for t in range(0, n_pos):
        m_new, max_val, f_act, i_act = _gates(i_ptr, f_ptr, t, str_t_if, head, m_old, DTYPE)
        v_val = tl.load(v_ptr + t * str_t_v + head * DV + offs_v)
        qk_off = t * str_t_qk + head * DQ
        k0 = tl.load(k_ptr + qk_off + offs_q)
        c0 = f_act * c0 + i_act * (k0[:, None] * v_val[None, :])
        n0 = f_act * n0 + i_act * k0
        if TWO:
            k1 = tl.load(k_ptr + qk_off + BQ + offs_q)
            c1 = f_act * c1 + i_act * (k1[:, None] * v_val[None, :])
            n1 = f_act * n1 + i_act * k1
        m_old = m_new
    tl.store(c_base + offs_q[:, None] * DV + offs_v[None, :], c0)
    if TWO:
        tl.store(c_base + (BQ + offs_q[:, None]) * DV + offs_v[None, :], c1)
    if idx_v == 0:
        tl.store(n_ptr + head * DQ + offs_q, n0)
        if TWO:
            tl.store(n_ptr + head * DQ + BQ + offs_q, n1)
        tl.store(m_ptr + head, m_old)


@triton.jit
def accept_kernel(ids_ptr, tok_ptr, out_ptr, S: tl.constexpr, SMAX: tl.constexpr):
    """out[:S] <- tok[:S]; out[SMAX] <- L = the number of leading drafts ids[1+j] equal to tok[j] (j < S - 1)."""
    alive = (tl.load(ids_ptr) == tl.load(ids_ptr)).to(tl.int32)  # 1, as a tensor (loop-carried)
    n_acc = alive * 0
    for j in range(0, S - 1):
        alive = alive * (tl.load(ids_ptr + 1 + j) == tl.load(tok_ptr + j)).to(tl.int32)
        n_acc += alive
    for j in range(0, S):
        tl.store(out_ptr + j, tl.load(tok_ptr + j))
    tl.store(out_ptr + SMAX, n_acc.to(tl.int64))
