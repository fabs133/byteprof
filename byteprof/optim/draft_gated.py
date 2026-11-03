"""Boundary-gated lookup drafting (P4, TASK_build_p4.md): host loop and the pieces shared with the CPU toy test.

Rounds that start mid-patch (the last emitted atom carries no boundary flag) are today's within-patch rounds
(`draft_core.rounds_loop` semantics: byte rule, a flagged prediction ends the round, no global pass). Rounds that start
at a patch boundary draft across patches: the lookup draft gets boundary flags (dolma2 re-tokenization of context +
draft, the `cut` protocol of the crossing task), the encoder runs over the K + 1 positions, one global pass runs over
the P speculative patches (a0's and one per flagged draft atom), the decoder picks the patch vector of every position
and the draft is accepted while byte *and* flag match (raw ids, flag included).

Per round (S = K + 1 positions, f_j = boundary flag of the input atom at position j, f_0 = 1):
  patch_select[j] = (f_0 + .. + f_j) - 1      the speculative patch whose vector the decoder uses at position j
  pool[r]         = position of the r-th flagged input   (the encoder output that becomes patch r)
  accept          L = longest prefix with draft_raw[j] == p_j   (raw ids carry the flag)
  rollback        patches kept = patch_select[L] + 1; the global KV position and the last patch vector follow it

`gated_rounds_loop` is the host loop; the device pieces (`select_patch_values`, `rollback_index`, `accepted_raw`) are
small torch functions used by both the CUDA-graph engine and the toy model, so the toy test exercises them.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..drafting import flag_of
from ..drafting.diagnostics import limit_patches
from . import draft_core as core

PMAX = 8  # most speculative patches per crossing round (a0's included)


# ---------------------------------------------------------------------------------------
# host: the plan of a crossing round
# ---------------------------------------------------------------------------------------


@dataclass
class CrossPlan:
    ids: list[int]  # plain ids of the inputs: [byte(a0), d_1 .. d_K]
    raw_ids: list[int]  # draft ids with the flag (+ boundary_offset when flagged); position 0 repeats the plain a0
    subs: list[int]
    patch_select: list[int]  # per position, the patch whose vector the decoder uses
    pool: list[int]  # positions whose encoder output becomes a patch
    states: list[int]  # automaton state before each position's successor (as in rounds_loop)

    @property
    def n_patches(self) -> int:
        return len(self.pool)


def patch_select(flags_in: Sequence[int]) -> list[int]:
    out, n = [], 0
    for f in flags_in:
        n += int(bool(f))
        out.append(n - 1)
    return out


def pool_positions(flags_in: Sequence[int]) -> list[int]:
    return [j for j, f in enumerate(flags_in) if f]


def raw_draft_ids(draft: Sequence[int], byte_offset: int, boundary_offset: int) -> list[int]:
    return [(a & 255) + byte_offset + (boundary_offset if flag_of(a) else 0) for a in draft]


def plan_crossing(raw: int, state: int, sub: int, draft: Sequence[int], automaton: Any, byte_offset: int,
                  boundary_offset: int) -> CrossPlan:
    """Inputs of a crossing round whose a0 = `raw` (flagged) is followed by the flagged `draft` atoms."""
    assert raw >= boundary_offset, "a crossing round starts at a boundary"
    a0_id = raw - boundary_offset
    ids, subs, states = [a0_id], [sub], [state]
    st = state
    for a in draft:
        did = (a & 255) + byte_offset
        st, sb = automaton.step(st, did)
        ids.append(did)
        subs.append(sb)
        states.append(st)
    flags_in = [1] + [flag_of(a) for a in draft]
    return CrossPlan(ids, [a0_id] + raw_draft_ids(draft, byte_offset, boundary_offset), subs,
                     patch_select(flags_in), pool_positions(flags_in), states)


# ---------------------------------------------------------------------------------------
# device pieces (torch; used inside the CUDA graphs and by the toy engine)
# ---------------------------------------------------------------------------------------


def select_patch_values(values: torch.Tensor, psel: torch.Tensor) -> torch.Tensor:
    """values (P, d) per patch, psel (S,) -> (S, d): the patch vector of every position."""
    return values.index_select(0, psel)


def rollback_index(psel: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """(1,) index of the last patch kept after accepting L drafts: the patch used at position L."""
    return psel.index_select(0, length.reshape(1))


def patches_kept(psel: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """(1,) number of speculative patches that stay in the global KV cache."""
    return rollback_index(psel, length) + 1


def accepted_raw(raw_ids: torch.Tensor, tok: torch.Tensor, s: int) -> torch.Tensor:
    """0-d L: leading drafts whose raw id (flag included) equals the prediction at the previous position."""
    return core.accepted_length_t(raw_ids, tok, s)


# ---------------------------------------------------------------------------------------
# host loop
# ---------------------------------------------------------------------------------------


class GatedStats(core.Stats):
    def __init__(self) -> None:
        super().__init__()
        self.kinds = {"within": 0, "cross": 0}
        self.kind_emitted = {"within": 0, "cross": 0}
        self.patches_hist: dict[int, int] = {}  # speculative patches of a crossing round -> rounds
        self.patches_kept = 0  # patches that went through the global model and stayed
        self.global_patches = 0  # patches run through the global model (speculative, incl. rejected)
        self.cap_cut = 0  # crossing drafts cut by the patch cap / the free global-cache slots
        self.retok_us = 0.0
        self.retok_calls = 0
        self.propose_us = 0.0

    def dump(self) -> dict:
        d = super().dump()
        d.update({"kinds": dict(self.kinds), "kind_emitted": dict(self.kind_emitted),
                  "patches_hist": dict(self.patches_hist), "patches_kept": self.patches_kept,
                  "global_patches": self.global_patches, "cap_cut": self.cap_cut,
                  "retok_us_per_call": self.retok_us / max(self.retok_calls, 1), "retok_calls": self.retok_calls,
                  "propose_us_per_round": self.propose_us / max(self.rounds, 1)})
        return d


Flagger = Callable[[bytes, list[int], int], list[int]]


def gated_rounds_loop(engine: Any, automaton: Any, drafter: Any, *, state: int, sub: int, raw: int, tokens: list[int],
                      n_tok: int, stop_len: int, eos_ids: tuple[int, ...], bpe_end: int, byte_offset: int,
                      boundary_offset: int, k_max: int, k_cross: int, max_patches: int, pos: int, capacity: int,
                      ctx: bytearray, flagger: Flagger | None = None, stats: GatedStats | None = None,
                      on_round: Callable[[dict], None] | None = None) -> int:
    """`rounds_loop` with boundary-gated rounds. `pos`: patches already in the global KV cache, `capacity` its size,
    `ctx`: the bytes of the context including a0 (extended here). Returns n_tok."""
    ks = engine.k_values
    cap_patches = min(max_patches, PMAX)
    while True:
        boundary = raw >= boundary_offset
        if raw == bpe_end:
            raise RuntimeError("model produced <bpe_token_end>; unsupported at batch 1")
        if boundary and pos >= capacity:
            raise RuntimeError(f"the static global cache is full ({capacity} patches)")
        cross_try = boundary and k_cross > 0 and cap_patches > 1 and len(ks) > 1
        kd = k_cross if cross_try else k_max
        t0 = time.perf_counter()
        draft = drafter.propose(kd) if (drafter is not None and kd > 0 and len(ks) > 1) else []
        t1 = time.perf_counter()
        if cross_try and draft and flagger is not None:
            draft = flagger(bytes(ctx), list(draft), kd)
            if stats is not None:
                stats.retok_us += (time.perf_counter() - t1) * 1e6
                stats.retok_calls += 1
        if cross_try:
            slots = min(cap_patches, capacity - pos)
            draft, cut = limit_patches(list(draft), 1, slots)
            if stats is not None:
                stats.cap_cut += int(cut)
        if stats is not None:
            stats.propose_us += (time.perf_counter() - t0) * 1e6
        k = core.choose_k(len(draft), ks, kd)
        draft = list(draft[:k])
        n_p = 1 + sum(flag_of(a) for a in draft) if cross_try else int(boundary)
        crossing = cross_try and k > 0 and n_p > 1
        if crossing:
            plan = plan_crossing(raw, state, sub, draft, automaton, byte_offset, boundary_offset)
            preds, l_dev = engine.run_cross(k, plan.ids, plan.raw_ids, plan.subs, plan.patch_select, plan.pool)
            L = core.accepted_length(plan.raw_ids, preds, k)
            ids, states = plan.ids, plan.states
            kept = plan.patch_select[L] + 1
        else:
            a0_id = raw - boundary_offset if boundary else raw
            ids, subs, states = [a0_id], [sub], [state]
            st = state
            for j in range(k):
                did = (draft[j] & 255) + byte_offset
                st, sb = automaton.step(st, did)
                ids.append(did)
                subs.append(sb)
                states.append(st)
            preds, l_dev = engine.run_round(k, boundary, ids, subs)
            L = core.accepted_length(ids, preds, k)
            kept = int(boundary)
        if L != l_dev:
            raise RuntimeError(f"host and device disagree on the accepted length ({L} vs {l_dev})")
        pos += kept
        emitted = [int(p) for p in preds[:L + 1]]
        kind = "cross" if crossing else "within"
        if stats is not None:
            stats.rounds += 1
            stats.by_k[k] = stats.by_k.get(k, 0) + 1
            stats.l_hist[L] = stats.l_hist.get(L, 0) + 1
            stats.global_calls += int(boundary)
            stats.kinds[kind] += 1
            stats.patches_kept += kept
            stats.global_patches += n_p if crossing else int(boundary)
            if crossing:
                stats.patches_hist[n_p] = stats.patches_hist.get(n_p, 0) + 1
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
            stats.kind_emitted[kind] += take
        if on_round is not None:
            on_round({"k": k, "L": L, "boundary": boundary, "emitted": emitted, "ids": ids, "preds": list(preds),
                      "crossing": crossing, "patches": n_p, "kept": kept, "pos": pos})
        if done:
            return n_tok
        atoms = [core.atom_of(r, byte_offset, boundary_offset) for r in emitted]
        atoms = [a for a in atoms if a is not None]
        if drafter is not None:
            drafter.push(atoms)
        ctx.extend(a & 255 for a in atoms)
        raw = emitted[-1]
        state, sub = automaton.step(states[L], raw)


def make_flagger(dol: Any) -> Flagger:
    """Lookup draft -> draft with dolma2 boundary flags (the `cut` protocol of `RecordedSources.lookup_flagged`)."""

    def flagger(ctx: bytes, draft: list[int], k: int) -> list[int]:
        from ..drafting.lmdraft import Cont, build_draft

        cont = bytes(a & 255 for a in draft)
        return build_draft(ctx, Cont(0, cont, [len(cont)]), k, dol, "cut")[0]

    return flagger
