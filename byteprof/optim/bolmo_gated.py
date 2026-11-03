"""Crossing rounds of the boundary-gated policy for Bolmo (TASK_build_p4.md): extra CUDA graphs next to the K graphs of
`bolmo_draft.DraftEngine`.

A crossing round starts at a patch boundary and drafts across patches. Its graphs, for K drafted bytes and P speculative
patches (P = 1 + the number of flagged drafted atoms, 2 <= P <= 8):

  encd{K}   the existing encoder over S = K + 1 positions (unchanged)
  gx{P}     pooling: the encoder outputs at the P flagged inputs -> out_projection -> the 16 global layers over P rows
            (one pass, static KV cache, rows pos .. pos + P - 1), patch vectors into `gm_out[:, :P]`; `glob.pos` is
            not advanced here
  decx{K}   the decoder over S positions, each with the vector of its patch (`patch_select`), lm_head, acceptance on raw
            ids (byte and flag), the state commit of the mLSTM layers (Task A's lazy commit with `fused`), and the
            rollback: `glob.pos += patches kept`, `last_value` = the vector of the last kept patch

Host inputs go through one pinned buffer (`d_ids`): [plain ids | subword ids | raw ids | patch select | pool positions].
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import torch

from . import draft_gated as gated
from .bolmo_draft import DraftEngine


class GatedEngine(DraftEngine):
    def __init__(self, model: Any, loop: Any, glob: Any, pieces: Any, runner: Any, ks: tuple[int, ...], device: str,
                 verify: str = "steps"):
        super().__init__(model, loop, glob, pieces, runner, ks, device, verify)
        s = self.smax
        n = 4 * s + gated.PMAX
        self.o_raw, self.o_psel, self.o_pool = 2 * s, 3 * s, 4 * s
        self.d_ids = torch.zeros(n, dtype=torch.long, device=device)  # replaces the 2 * smax buffer before capture
        self.p_d = torch.zeros(n, dtype=torch.long).pin_memory()
        self.d_np = self.p_d.numpy()
        self.gm_out = torch.zeros(1, gated.PMAX, loop.d, dtype=self.dtype, device=device)
        self.ar = torch.arange(gated.PMAX, device=device)
        self.max_p = min(gated.PMAX, self.kmax + 1)

    # ---- pooling + the global model over P patches ------------------------------------------------
    def glob_x(self, p: int) -> None:
        m = self.glob.m
        pool = self.d_ids[self.o_pool:self.o_pool + p]
        rows = self.h_bytes[0].index_select(0, pool)  # (P, d): encoder outputs at the flagged inputs
        h = self.loop.enc.out_projection(rows.unsqueeze(0))
        cp = self.glob.pos + self.ar[:p]
        position_ids = cp.unsqueeze(0)
        pe = m.rotary_embs["full_attention"](h, position_ids)
        mask = (self.glob.kv_index[None, :] <= cp[:, None]).view(1, 1, p, -1)
        for layer in m.layers:
            h = layer(h, attention_mask=mask, position_ids=position_ids, past_key_values=self.glob.cache,
                      cache_position=cp, position_embeddings=pe)
        self.gm_out[:, :p].copy_(h)

    # ---- the multi-patch decoder, acceptance, commit, rollback --------------------------------------
    def dec_x(self, s: int) -> None:
        m = self.model_ref[0]
        dec = self.loop.dec
        psel = self.d_ids[self.o_psel:self.o_psel + s]
        hp = dec.initial_norm(self.gm_out)[0]  # (PMAX, d); rows past P are not selected
        vals = gated.select_patch_values(hp, psel)
        h = dec.in_projection(self.h_bytes[:, :s]) + vals.unsqueeze(0)
        n_enc = len(self.enc_layers)
        for li, layer in enumerate(self.dec_layers):
            h = self._layer(layer, h, n_enc + li, s)
        logits = m.lm_head(m.model.norm(h))
        tok = torch.argmax(logits, dim=-1)[0]
        raw_ids = self.d_ids[self.o_raw:self.o_raw + self.smax]
        if self.verify == "fused":
            from . import draft_fused

            draft_fused.accept(raw_ids, tok, self.out_dev, s, self.smax)
            length = self.out_dev[self.smax]
            for li in range(len(self.cur)):
                draft_fused.lazy_commit(self.cur[li], *self.saved[(s, li)], length, *self.nm_start[li])
        else:
            length = gated.accepted_raw(raw_ids, tok, s)
            self.out_dev[:s].copy_(tok)
            self.out_dev[self.smax].copy_(length)
            for li in range(len(self.cur)):
                from . import draft_core as core

                core.commit_state(self.hist[li], self.cur[li], length)
        last = gated.rollback_index(psel, length)
        self.loop.last_value.copy_(hp.index_select(0, last))
        self.glob.pos.add_(last + 1)

    def _extra_fns(self) -> list[tuple[str, Callable]]:
        fns: list[tuple[str, Callable]] = [(f"gx{p}", partial(self.glob_x, p)) for p in range(1, self.max_p + 1)]
        fns += [(f"decx{k}", partial(self.dec_x, k + 1)) for k in self.k_values if k > 0]
        return fns

    # ---- one crossing round ------------------------------------------------------------------------
    def run_cross(self, k: int, ids: Any, raw_ids: Any, subs: Any, psel: Any, pool: Any) -> tuple[Any, int]:
        run = self.runner.run
        s, smax = k + 1, self.smax
        d = self.d_np
        d[:s] = ids
        d[smax:smax + s] = subs
        d[self.o_raw:self.o_raw + s] = raw_ids
        d[self.o_psel:self.o_psel + s] = psel
        d[self.o_pool:self.o_pool + len(pool)] = pool
        self.d_ids.copy_(self.p_d, non_blocking=True)
        run(f"encd{k}")
        run(f"gx{len(pool)}")
        run(f"decx{k}")
        self.p_out.copy_(self.out_dev, non_blocking=True)
        self.done.record()
        self.done.synchronize()
        return self.out_np[:s].tolist(), int(self.out_np[smax])
