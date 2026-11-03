"""Shared pieces of the drafting loop (TASK_drafting.md): used by Bolmo (`bolmo_draft.py`) and by the CPU toy
model of `tests/test_draft_toy.py`, so that the toy test exercises the same rollback / indexing code.

Round (TASK_drafting.md "The round rule"), with S = K + 1 positions:
  inputs      bytes[0..K] = [byte(a0), d_1 .. d_K], subs[0..K] = subword ids (automaton state advanced over the drafts)
  A_K         encoder over S positions; the recurrent state after every position goes into a history slot
  B           global model, only if a0 carried the boundary flag (on position 0 of the encoder output)
  C_K         decoder over S positions with the current patch vector, then lm_head: S predicted raw ids p_1 .. p_S
  accept      L = longest prefix with d_j == plain(p_j) (a boundary-flagged p_j never equals a plain byte id)
  commit      state after position L = history slot L -> the current state (an index, no recomputation)
  emitted     p_1 .. p_{L+1}
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import torch


# ---------------------------------------------------------------------------------------
# acceptance
# ---------------------------------------------------------------------------------------


def accepted_length(ids: Sequence[int], toks: Sequence[int], k: int) -> int:
    """ids = [byte(a0), d_1 .. d_K] (plain byte ids), toks = p_1 .. p_{K+1} raw ids. Returns L."""
    n = 0
    while n < k and ids[n + 1] == toks[n]:
        n += 1
    return n


def accepted_length_t(ids: torch.Tensor, tok: torch.Tensor, s: int) -> torch.Tensor:
    """Same on the device (0-d int32): ids (>= S,), tok (>= S,); drafts are ids[1:S] against tok[:S-1]."""
    return torch.cumprod((ids[1:s] == tok[:s - 1]).to(torch.int32), 0).sum()


# ---------------------------------------------------------------------------------------
# recurrence over S positions with a state history
# ---------------------------------------------------------------------------------------


def scan_with_history(step: Callable, cur: tuple[torch.Tensor, ...], hist: tuple[torch.Tensor, ...], s: int,
                      q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, i: torch.Tensor, f: torch.Tensor,
                      hbuf: torch.Tensor) -> None:
    """S sequential recurrence steps on precomputed projections.

    cur   (c, n, m): the state before position 0, shapes (1,NH,Dq,Dv), (1,NH,Dq), (1,NH,1)
    hist  (C, N, M): slots, shapes (Smax,1,NH,Dq,Dv), ...; slot j receives the state after position j
    q, k  (S,NH,Dq); v (S,NH,Dv); i, f (S,NH,1); hbuf (S,NH,Dv) receives the cell outputs.
    `step(prev, new, q1, k1, v1, i1, f1, h1)` is one recurrence step writing `new` and `h1` in place.
    """
    prev = cur
    for j in range(s):
        new = (hist[0][j], hist[1][j], hist[2][j])
        step(prev, new, q[j:j + 1], k[j:j + 1], v[j:j + 1], i[j:j + 1], f[j:j + 1], hbuf[j:j + 1])
        prev = new


def scan_without_history(step: Callable, cur: tuple[torch.Tensor, ...], s: int, q: torch.Tensor, k: torch.Tensor,
                         v: torch.Tensor, i: torch.Tensor, f: torch.Tensor, hbuf: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Lazy-commit variant of `scan_with_history` (reference for the fused kernel of TASK_parallel_verify.md): S steps
    from `cur` writing only the cell outputs; `cur` is not touched and no per-position state is kept. Returns a copy of
    the start state, which `lazy_commit_state` needs."""
    start = tuple(t.clone() for t in cur)
    a, b = tuple(torch.empty_like(t) for t in cur), tuple(torch.empty_like(t) for t in cur)
    prev = start
    for j in range(s):
        new = a if j % 2 == 0 else b
        step(prev, new, q[j:j + 1], k[j:j + 1], v[j:j + 1], i[j:j + 1], f[j:j + 1], hbuf[j:j + 1])
        prev = new
    return start


def lazy_commit_state(step: Callable, cur: tuple[torch.Tensor, ...], start: tuple[torch.Tensor, ...],
                      length: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, i: torch.Tensor,
                      f: torch.Tensor) -> None:
    """cur <- the state after positions 0 .. L, recomputed from `start` with the saved projections (L: 0-d tensor)."""
    a, b = tuple(torch.empty_like(t) for t in cur), tuple(torch.empty_like(t) for t in cur)
    h = torch.empty(1, *q.shape[1:-1], v.shape[-1], dtype=q.dtype)
    prev = start
    for j in range(int(length) + 1):
        new = a if j % 2 == 0 else b
        step(prev, new, q[j:j + 1], k[j:j + 1], v[j:j + 1], i[j:j + 1], f[j:j + 1], h)
        prev = new
    for dst, src in zip(cur, prev):
        dst.copy_(src)


def commit_state(hist: tuple[torch.Tensor, ...], cur: tuple[torch.Tensor, ...], length: torch.Tensor) -> None:
    """cur <- hist[length] (length: 0-d integer tensor on the same device)."""
    if cur[0].is_cuda:
        _commit_triton(hist, cur, length)
    else:
        idx = int(length)
        for h, c in zip(hist, cur):
            c.copy_(h[idx])


def _commit_triton(hist: tuple[torch.Tensor, ...], cur: tuple[torch.Tensor, ...], length: torch.Tensor) -> None:
    import triton

    from .draft_kernels import commit_kernel

    c_n, n_n, m_n = cur[0].numel(), cur[1].numel(), cur[2].numel()
    block = 1024
    assert n_n <= c_n and m_n <= c_n
    commit_kernel[(triton.cdiv(c_n, block),)](hist[0], cur[0], hist[1], cur[1], hist[2], cur[2], length,
                                              c_n, n_n, m_n, BLOCK=block)


def mlstm_step_into(prev: tuple[torch.Tensor, ...], new: tuple[torch.Tensor, ...], q: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor, i: torch.Tensor, f: torch.Tensor, h: torch.Tensor, eps: float) -> None:
    """mlstm_kernels' Triton recurrent step (the kernel the minimal loop uses), writing into given buffers.

    Same arguments as `mlstm_recurrent_step__triton_fw` computes them; it only avoids allocating the
    new state and the output, so the state after every position can go into a history slot."""
    import triton
    from mlstm_kernels.torch.utils import torch2triton_dtype
    from mlstm_kernels.triton.kernel_param_heuristics import get_head_dim_block_size
    from mlstm_kernels.triton.recurrent.fw_step_fused import recurrent_step_fw_kernel

    c_old, n_old, m_old = prev
    c_new, n_new, m_new = new
    b, nh, dhqk = q.shape
    dhhv = v.shape[-1]
    siz_b_dhqk = get_head_dim_block_size(head_dim=dhqk, min_block_size=64)
    siz_b_dhhv = get_head_dim_block_size(head_dim=dhhv, min_block_size=64)
    grid = (1, triton.cdiv(dhhv, siz_b_dhhv), b * nh)
    recurrent_step_fw_kernel[grid](
        matC_old=c_old, vecN_old=n_old, scaM_old=m_old, vecQ=q, vecK=k, vecV=v, scaI=i, scaF=f, vecH=h,
        matC_new=c_new, vecN_new=n_new, scaM_new=m_new, qk_scale=dhqk ** -0.5,
        str_matC_B_NH=c_old.stride(1), str_matC_DHQK=c_old.stride(2), str_matC_DHHV=c_old.stride(3),
        str_vecN_B_NH=n_old.stride(1), str_vecN_DHQK=n_old.stride(2), str_scaM_B_NH=m_old.stride(1),
        str_vecQK_NH=q.stride(1), str_vecQK_DHQK=q.stride(2), str_vecVH_B_NH=v.stride(1),
        str_vecVH_DHHV=v.stride(2), str_scaIF_B_NH=i.stride(1),
        B=b, NH=nh, DHQK=dhqk, DHHV=dhhv, EPS=eps, DTYPE=torch2triton_dtype(q.dtype),
        DTYPE_STATE=torch2triton_dtype(c_old.dtype), siz_b_DHQK=siz_b_dhqk, siz_b_DHHV=siz_b_dhhv,
        num_warps=4 if siz_b_dhqk >= 64 else 2, num_stages=1)


# ---------------------------------------------------------------------------------------
# host round loop
# ---------------------------------------------------------------------------------------


class RoundEngine(Protocol):
    k_values: tuple[int, ...]  # K values with a graph / implementation, always including 0

    def run_round(self, k: int, boundary: bool, ids: Sequence[int], subs: Sequence[int]) -> tuple[Sequence[int], int]:
        """Run A_K, B (if `boundary`) and C_K for the K + 1 inputs; return (the K + 1 predicted raw ids, L as
        computed by the device)."""


class Automaton(Protocol):
    def step(self, state: int, byte: int) -> tuple[int, int]: ...


class Stats:
    def __init__(self) -> None:
        self.rounds = 0
        self.emitted = 0
        self.by_k: dict[int, int] = {}
        self.l_hist: dict[int, int] = {}
        self.global_calls = 0
        self.draft_us = 0.0

    def dump(self) -> dict:
        return {"rounds": self.rounds, "emitted": self.emitted, "by_k": dict(self.by_k), "l_hist": dict(self.l_hist),
                "global_calls": self.global_calls, "bytes_per_round": self.emitted / max(self.rounds, 1)}


def atom_of(raw: int, byte_offset: int, boundary_offset: int) -> int | None:
    flag = raw >= boundary_offset
    plain = raw - boundary_offset if flag else raw
    b = plain - byte_offset
    return (b | (int(flag) << 8)) if 0 <= b < 256 else None


def choose_k(draft_len: int, k_values: Sequence[int], k_cap: int) -> int:
    """Largest available K not above the draft length and the cap."""
    best = 0
    for kv in k_values:
        if kv <= draft_len and kv <= k_cap and kv > best:
            best = kv
    return best


def rounds_loop(engine: RoundEngine, automaton: Automaton, drafter: Any, *, state: int, sub: int, raw: int,
                tokens: list[int], n_tok: int, stop_len: int, eos_ids: tuple[int, ...], bpe_end: int,
                byte_offset: int, boundary_offset: int, k_max: int, stats: Stats | None = None,
                on_round: Callable[[dict], None] | None = None, k_cap_for: Callable[[Any], int] | None = None) -> int:
    """Rounds until `stop_len` atoms or EOS. `raw` is the last emitted atom a0 (its automaton `state` / `sub` are
    given). `tokens` receives stripped plain ids at tokens[n_tok:], as the minimal loop does. Returns n_tok."""
    ks = engine.k_values
    while True:
        boundary = raw >= boundary_offset
        if raw == bpe_end:
            raise RuntimeError("model produced <bpe_token_end>; unsupported at batch 1")
        draft = drafter.propose(k_max) if (drafter is not None and k_max > 0 and len(ks) > 1) else []
        cap = k_max if k_cap_for is None else k_cap_for(drafter)
        k = choose_k(len(draft), ks, cap)
        a0_id = raw - boundary_offset if boundary else raw
        ids = [a0_id]
        subs = [sub]
        states = [state]
        st = state
        for j in range(k):
            did = (draft[j] & 255) + byte_offset
            st, sb = automaton.step(st, did)
            ids.append(did)
            subs.append(sb)
            states.append(st)
        preds, l_dev = engine.run_round(k, boundary, ids, subs)
        L = accepted_length(ids, preds, k)
        if L != l_dev:
            raise RuntimeError(f"host and device disagree on the accepted length ({L} vs {l_dev})")
        emitted = [int(p) for p in preds[:L + 1]]
        if stats is not None:
            stats.rounds += 1
            stats.by_k[k] = stats.by_k.get(k, 0) + 1
            stats.l_hist[L] = stats.l_hist.get(L, 0) + 1
            stats.global_calls += int(boundary)
        done = False
        take = 0
        for r in emitted:
            if r == bpe_end:
                raise RuntimeError("model produced <bpe_token_end>; unsupported at batch 1")
            tokens[n_tok] = r - boundary_offset if r >= boundary_offset else r
            n_tok += 1
            take += 1
            if r in eos_ids or n_tok >= stop_len:
                done = True
                break
        emitted = emitted[:take]
        if stats is not None:
            stats.emitted += take
        if on_round is not None:
            on_round({"k": k, "L": L, "boundary": boundary, "emitted": emitted, "ids": ids, "preds": list(preds)})
        if done:
            return n_tok
        if drafter is not None:
            atoms = [atom_of(r, byte_offset, boundary_offset) for r in emitted]
            drafter.push([a for a in atoms if a is not None])
        raw = emitted[-1]
        state, sub = automaton.step(states[L], raw)
