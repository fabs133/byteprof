"""Patch-level drafting for Bolmo (batch 1, greedy): verify K drafted bytes per local pass.

Builds on the minimal loop (`bolmo_loop.py`, left untouched; its prefill is reproduced below):
`A_K` (encoder over K + 1 positions), `B` (the existing global graph), `C_K` (decoder over K + 1 positions +
norm + lm_head + argmax + acceptance + state commit) are graphed once per K. Projections, MLPs and norms run
batched over the K + 1 rows; the mLSTM recurrence runs as K + 1 sequential steps of the same Triton step
kernel, each writing its state into a history slot, and the device commits the slot of the last accepted
position (`draft_core.commit_state`). One host sync per round. See draft_core.py for the round rule.
"""

from __future__ import annotations

import sys
from functools import partial
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from . import OptimizeError
from . import draft_core as core


def _soft_cap(x: torch.Tensor, cap: float | None) -> torch.Tensor:
    return x if cap is None else cap * torch.tanh(x / cap)


class DraftEngine(nn.Module):
    """K + 1 position pieces for the local encoder and decoder, graphed per K."""

    def __init__(self, model: Any, loop: Any, glob: Any, pieces: Any, runner: Any, ks: tuple[int, ...], device: str,
                 verify: str = "steps"):
        super().__init__()
        self.verify = verify
        self.model_ref = [model]
        self.loop, self.glob, self.pieces, self.runner = loop, glob, pieces, runner
        self.k_values = (0,) + tuple(sorted(k for k in ks if k > 0))
        self.kmax = max(self.k_values)
        self.smax = self.kmax + 1
        self.device = device
        self.dtype = loop.h_byte.dtype
        d = loop.d
        self.enc_layers = list(loop.enc.layers)
        self.dec_layers = list(loop.dec.layers)
        self.cur = list(loop.enc_states) + list(loop.dec_states)
        self.hist: list[tuple[torch.Tensor, ...]] = []
        self.nm_start: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.saved: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
        self.hbuf: list[torch.Tensor] = []
        for layer, st in zip(self.enc_layers + self.dec_layers, self.cur):
            xl = layer.xlstm
            nh = xl.config.num_heads
            if verify == "steps":
                self.hist.append(tuple(torch.zeros(self.smax, *t.shape, dtype=t.dtype, device=device) for t in st))
            else:  # fused: the start n, m for the lazy commit; k, v, gates of the first pass are kept per graph
                self.nm_start.append((torch.zeros(nh, st[1].shape[-1], dtype=st[1].dtype, device=device),
                                      torch.zeros(nh, dtype=st[2].dtype, device=device)))
            self.hbuf.append(torch.zeros(self.smax, nh, xl.v_dim // nh, dtype=self.dtype, device=device))
        self.h_bytes = torch.zeros(1, self.smax, d, dtype=self.dtype, device=device)
        self.d_ids = torch.zeros(2 * self.smax, dtype=torch.long, device=device)  # [bytes | subword ids]
        self.out_dev = torch.zeros(self.smax + 1, dtype=torch.long, device=device)  # [p_1 .. p_S, ..., L]
        self.p_d = torch.zeros(2 * self.smax, dtype=torch.long).pin_memory()
        self.p_out = torch.zeros(self.smax + 1, dtype=torch.long).pin_memory()
        self.p_ids = torch.zeros(2, dtype=torch.long).pin_memory()
        self.p_tok = torch.zeros(1, dtype=torch.long).pin_memory()
        self.d_np, self.out_np = self.p_d.numpy(), self.p_out.numpy()
        self.ids_np, self.tok_np = self.p_ids.numpy(), self.p_tok.numpy()
        self.done = torch.cuda.Event()
        self.capture_seconds = 0.0

    # ---- the K + 1 position layer ---------------------------------------------------------
    def _layer(self, layer: Any, h: torch.Tensor, li: int, s: int) -> torch.Tensor:
        xl = layer.xlstm
        cfg = xl.config
        nh = cfg.num_heads
        x = layer.pre_xlstm_layernorm(h)
        q, k, v, o_pre = xl.q(x), xl.k(x), xl.v(x), xl.ogate_preact(x)
        i_pre = _soft_cap(xl.igate_preact(x), cfg.gate_soft_cap)
        f_pre = _soft_cap(xl.fgate_preact(x), cfg.gate_soft_cap)
        dq, dv = q.shape[-1] // nh, v.shape[-1] // nh
        eps = xl.mlstm_backend.config.eps
        if self.verify == "fused":
            from . import draft_fused

            k3, v3, i3, f3 = k.view(s, nh, dq), v.view(s, nh, dv), i_pre.view(s, nh, 1), f_pre.view(s, nh, 1)
            draft_fused.multistep_fw(self.cur[li], q.view(s, nh, dq), k3, v3, i3, f3, self.hbuf[li],
                                     *self.nm_start[li], eps=eps)
            self.saved[(s, li)] = (k3, v3, i3, f3)  # read again by the lazy commit at the end of C_K
        else:
            core.scan_with_history(
                partial(core.mlstm_step_into, eps=eps), self.cur[li], self.hist[li], s,
                q.view(s, nh, dq), k.view(s, nh, dq), v.view(s, nh, dv), i_pre.view(s, nh, 1), f_pre.view(s, nh, 1),
                self.hbuf[li])
        h_norm = xl.multihead_norm(self.hbuf[li][:s].view(1, s, nh, dv)).reshape(1, s, -1)
        y = xl.out_proj(xl.ogate_act_fn(o_pre) * h_norm)
        h = h + y
        return h + layer.mlp(layer.pre_feedforward_layernorm(h))

    # A_K: encoder over S = K + 1 positions ---------------------------------------------------
    def enc_k(self, s: int) -> None:
        enc = self.loop.enc
        h = enc._embed(self.d_ids[:s].view(1, s), self.d_ids[self.smax:self.smax + s].view(1, s))
        for li, layer in enumerate(self.enc_layers):
            h = self._layer(layer, h, li, s)
        h = enc.post_last_block_norm(h)
        self.h_bytes[:, :s].copy_(h)
        self.loop.patch_out.copy_(enc.out_projection(h[:, :1]))  # patch vector from position 0 (as in A_0)

    # C_K: decoder over S positions + lm_head + acceptance + commit -------------------------------
    def dec_k(self, s: int, new_patch: bool) -> None:
        m = self.model_ref[0]
        dec = self.loop.dec
        if new_patch:
            h_patch = dec.initial_norm(self.glob.glob_out)[:, -1:, :]
            self.loop.last_value.copy_(h_patch[:, -1])
        h = dec.in_projection(self.h_bytes[:, :s]) + self.loop.last_value.unsqueeze(1)
        n_enc = len(self.enc_layers)
        for li, layer in enumerate(self.dec_layers):
            h = self._layer(layer, h, n_enc + li, s)
        logits = m.lm_head(m.model.norm(h))
        tok = torch.argmax(logits, dim=-1)[0]
        if self.verify == "fused":
            from . import draft_fused

            draft_fused.accept(self.d_ids, tok, self.out_dev, s, self.smax)
            length = self.out_dev[self.smax]
            for li in range(len(self.cur)):
                draft_fused.lazy_commit(self.cur[li], *self.saved[(s, li)], length, *self.nm_start[li])
            return
        length = core.accepted_length_t(self.d_ids, tok, s)
        self.out_dev[:s].copy_(tok)
        self.out_dev[self.smax].copy_(length)
        for li in range(len(self.cur)):
            core.commit_state(self.hist[li], self.cur[li], length)

    # ---- graph capture --------------------------------------------------------------------
    def _extra_fns(self) -> list[tuple[str, Callable]]:
        """Further graphs captured with the K graphs (the gated policy adds its crossing graphs)."""
        return []

    def capture(self) -> None:
        import time

        t0 = time.perf_counter()
        torch.cuda.synchronize()
        fns: list[tuple[str, Callable]] = []
        for k in self.k_values:
            if k == 0:
                continue
            s = k + 1
            fns += [(f"encd{k}", partial(self.enc_k, s)), (f"decd{k}n", partial(self.dec_k, s, False)),
                    (f"decd{k}p", partial(self.dec_k, s, True))]
        fns += self._extra_fns()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                self.glob.step()
                for _, fn in fns:
                    fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.glob.pos.zero_()
        for name, fn in fns:
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g):
                fn()
            self.runner.graphs[name] = g
        torch.cuda.synchronize()
        self.capture_seconds = time.perf_counter() - t0

    # ---- one round ---------------------------------------------------------------------------
    def run_round(self, k: int, boundary: bool, ids: Any, subs: Any) -> tuple[Any, int]:
        run = self.runner.run
        if k == 0:
            self.ids_np[0] = ids[0]
            self.ids_np[1] = subs[0]
            self.pieces.ids.copy_(self.p_ids, non_blocking=True)
            run("enc_m")
            if boundary:
                run("glob")
                run("dec_p")
            else:
                run("dec_n")
            self.p_tok.copy_(self.pieces.tok, non_blocking=True)
            self.done.record()
            self.done.synchronize()
            return [int(self.tok_np[0])], 0
        s = k + 1
        self.d_np[:s] = ids
        self.d_np[self.smax:self.smax + s] = subs
        self.d_ids.copy_(self.p_d, non_blocking=True)
        run(f"encd{k}")
        if boundary:
            run("glob")
            run(f"decd{k}p")
        else:
            run(f"decd{k}n")
        self.p_out.copy_(self.out_dev, non_blocking=True)
        self.done.record()
        self.done.synchronize()
        return self.out_np[:s].tolist(), int(self.out_np[self.smax])


def build_drafter(spec: dict[str, Any], ngram: Any = None, truth: Any = None) -> Any:
    from ..drafting.drafters import CombinedDrafter, ContextDrafter, NullDrafter, OracleDrafter
    from ..drafting.ngram import NGramDrafter

    kind = spec.get("kind", "none")
    min_len = spec.get("min_len", 3)
    if kind == "none":
        return NullDrafter()
    if kind == "context":
        return ContextDrafter(min_len=min_len)
    if kind == "oracle":
        return OracleDrafter()
    ng = NGramDrafter(ngram, min_count=spec.get("min_count", 1), max_order=spec.get("max_order", 6))
    if kind == "ngram":
        return ng
    if kind == "combined":
        return CombinedDrafter(ContextDrafter(min_len=min_len), ng)
    raise OptimizeError(f"unknown draft kind {kind!r}")


def make_generate(model: Any, tok: Any, runner: Any, loop: Any, glob: Any, pieces: Any, automaton: Any,
                  engine: DraftEngine, spec: dict[str, Any], *, ngram: Any = None,
                  oracle_truth: dict[tuple[int, ...], list[int]] | None = None,
                  stats: core.Stats | None = None, on_round: Callable[[dict], None] | None = None) -> Callable:
    """Batch-1 greedy `generate()` with drafting. The prefill part is the minimal loop's (bolmo_loop.make_generate)."""
    from ..drafting import make_atom

    mod = sys.modules[type(model).__module__]
    MaskState = mod.MaskState
    bm = model.model
    enc, dec = bm.local_encoder, bm.local_decoder
    tk = bm.tokenizer
    boundary_offset = tk.offset + 256  # the model's fused-boundary offset (260)
    byte_offset = tk.offset
    eos, pad, bpe_end = tk.eos_token_id, tk.pad_token_id, tk.bpe_token_end_id
    dev = next(model.parameters()).device
    k_max = int(spec.get("k_max", engine.kmax))
    drafter = build_drafter(spec, ngram)
    flagger = None
    if spec.get("policy", "within") == "gated":
        from ..drafting.flags import Dolma2
        from . import draft_gated

        flagger = draft_gated.make_flagger(Dolma2())

    @torch.no_grad()
    def generate(inputs: torch.Tensor, generation_config: Any = None, logits_processor: Any = None,
                 stopping_criteria: Any = None, use_model_defaults: Any = None, **kwargs: Any) -> torch.Tensor:
        if inputs.shape[0] != 1 or kwargs.get("do_sample"):
            raise OptimizeError("the drafting loop supports batch size 1 and greedy decoding only")
        max_new = kwargs.get("max_new_tokens")
        if max_new is None:
            raise OptimizeError("the drafting loop needs max_new_tokens")
        prompt_len = inputs.shape[1]
        stop_len = (prompt_len - 1) + max_new
        for crit in stopping_criteria or []:
            n = getattr(crit, "n", None)
            if n is None:
                raise OptimizeError(f"unsupported stopping criterion {type(crit).__name__}")
            stop_len = min(stop_len, n + 2)

        # ---- prefill: same calls as bolmo_loop.make_generate (and the remote generate()) ------------
        expanded = torch.tensor(tk.expand_byte_ids(inputs[0].tolist()), device=dev, dtype=torch.long)[None]
        byte_input_ids = inputs
        sequence_start_indices = (byte_input_ids == pad).sum(-1)
        enc.free_inference_cache()
        dec.free_inference_cache()
        boundary_mask = bm.prefill_boundary_prediction_forward(
            byte_input_ids, expanded_input_ids=expanded, sequence_start_indices=sequence_start_indices)
        enc.prepare_inference_cache(1)
        dec.prepare_inference_cache(1)
        boundary_mask = boundary_mask[:, :-1]
        forced_id = byte_input_ids[:, -1].cpu().tolist()[0]
        byte_input_ids = byte_input_ids[:, :-1]
        sequence_start_indices = (byte_input_ids == pad).sum(-1)
        boundary_state = MaskState(boundary_mask[:, -1].clone())
        pad_state = MaskState(torch.zeros(1, dtype=torch.bool, device=dev))
        history = byte_input_ids[0].tolist()
        tokens = [0] * (stop_len + 2)
        tokens[0] = history[-1]
        n_tok = 1

        expanded_first = torch.tensor(tk.expand_byte_ids(history, n_last=len(history)), device=dev,
                                      dtype=torch.long)[None]
        out = model.forward(byte_input_ids, expanded_input_ids=expanded_first, boundary_mask=boundary_mask,
                            boundary_state=boundary_state, pad_state=pad_state,
                            sequence_start_indices=sequence_start_indices, logits_to_keep=1, use_cache=True,
                            past_key_values=None)
        logits = out.logits
        no_b = logits[0, 0, forced_id].item()
        b = logits[0, 0, forced_id + boundary_offset].item()
        logits[0, 0, :] = -100_000
        logits[0, 0, forced_id] = no_b
        logits[0, 0, forced_id + boundary_offset] = b
        raw = int(torch.argmax(logits[:, -1], dim=-1).item())

        def strip(r: int) -> int:
            return r - boundary_offset if r >= boundary_offset else r

        tokens[n_tok] = strip(raw)
        n_tok += 1
        finished = raw in (eos, eos + boundary_offset) or n_tok >= stop_len

        if not finished:
            for ls, bufs in zip(enc.layer_states, loop.enc_states):
                for dst, src in zip(bufs, ls["xlstm"]["state"]):
                    dst.copy_(src)
            for ls, bufs in zip(dec.layer_states, loop.dec_states):
                for dst, src in zip(bufs, ls["xlstm"]["state"]):
                    dst.copy_(src)
            loop.last_value.copy_(dec.last_value)
            glob.load_prefill(out.past_key_values)

            state = 0
            for h in history:
                state, _ = automaton.step(state, h)
            state, sub = automaton.step(state, raw)

            ctx = [make_atom(i - byte_offset, m) for i, m in zip(history, boundary_mask[0].tolist())
                   if byte_offset <= i < byte_offset + 256]
            a0 = core.atom_of(raw, byte_offset, boundary_offset)
            truth = None
            if oracle_truth is not None:
                truth = oracle_truth.get(tuple(inputs[0].tolist()))
                if truth is None:
                    raise OptimizeError("oracle drafter: no recorded ground truth for this prompt")
            drafter.reset(ctx + ([a0] if a0 is not None else []), truth)
            if spec.get("policy", "within") == "gated":
                from . import draft_gated

                ctx_bytes = bytearray(a & 255 for a in ctx) + bytearray(a0 & 255 for a0 in ([a0] if a0 is not None else []))
                n_tok = draft_gated.gated_rounds_loop(
                    engine, automaton, drafter, state=state, sub=sub, raw=raw, tokens=tokens, n_tok=n_tok,
                    stop_len=stop_len, eos_ids=(eos, eos + boundary_offset), bpe_end=bpe_end, byte_offset=byte_offset,
                    boundary_offset=boundary_offset, k_max=k_max, k_cross=int(spec.get("k_cross", 32)),
                    max_patches=int(spec.get("max_patches", 8)), pos=int(glob.pos.item()), capacity=glob.MAX_PATCHES,
                    ctx=ctx_bytes, flagger=flagger, stats=stats, on_round=on_round)
            else:
                n_tok = core.rounds_loop(
                    engine, automaton, drafter, state=state, sub=sub, raw=raw, tokens=tokens, n_tok=n_tok,
                    stop_len=stop_len, eos_ids=(eos, eos + boundary_offset), bpe_end=bpe_end, byte_offset=byte_offset,
                    boundary_offset=boundary_offset, k_max=k_max, stats=stats, on_round=on_round,
                    k_cap_for=_k_cap(spec, drafter, k_max))

        gen = torch.tensor(tokens[:n_tok], dtype=torch.long, device=dev)
        return torch.cat([byte_input_ids[0, :-1], gen]).unsqueeze(0)

    generate.drafter = drafter  # type: ignore[attr-defined]
    return generate


def _k_cap(spec: dict[str, Any], drafter: Any, k_max: int) -> Callable[[Any], int] | None:
    """Optional adaptive K: `k_ngram` caps K when the combined drafter answered from the n-gram."""
    k_ng = spec.get("k_ngram")
    if k_ng is None or not hasattr(drafter, "context"):
        return None
    last = {"ctx": 0}

    def cap(d: Any) -> int:
        used_ctx = d.used_context > last["ctx"]
        last["ctx"] = d.used_context
        return k_max if used_ctx else min(k_max, int(k_ng))

    return cap


def install(model: Any, tok: Any, cfg: Any, runner: Any, loop: Any, glob: Any, pieces: Any, automaton: Any) -> dict:
    """Called from bolmo.apply when [draft] kind != none. Returns what to attach to the handle."""
    spec = dict(cfg.draft)
    ks = tuple(spec.get("ks", (0, 1, 2, 4, 6, 8)))
    k_max = int(spec.get("k_max", max(ks)))
    k_top = max(k_max, int(spec.get("k_cross", 0))) if spec.get("policy", "within") == "gated" else k_max
    ks = tuple(k for k in ks if k <= k_top)
    if spec.get("policy", "within") == "gated":
        from . import bolmo_gated

        engine = bolmo_gated.GatedEngine(model, loop, glob, pieces, runner, ks, cfg.device,
                                         verify=spec.get("verify", "steps"))
    else:
        engine = DraftEngine(model, loop, glob, pieces, runner, ks, cfg.device, verify=spec.get("verify", "steps"))
    with torch.inference_mode():
        engine.capture()
    ngram = None
    if spec["kind"] in ("ngram", "combined"):
        from pathlib import Path

        from ..drafting.ngram import NGramModel

        cdir = Path(spec.get("corpus_dir", "results/drafting/corpus"))
        data = (cdir / "corpus.bin").read_bytes()
        flags = np.fromfile(cdir / "flags.bin", np.uint8)
        ngram = NGramModel.build(data, flags, tuple(spec.get("orders", (1, 2, 3, 4, 5, 6, 8, 12))), bits=24)
    oracle = None
    if spec["kind"] == "oracle":
        from ..drafting.truthio import atom_to_raw  # noqa: F401
        import json
        from pathlib import Path

        obj = json.loads(Path(spec["truth_file"]).read_text(encoding="utf-8"))
        n = str(spec.get("truth_length", max(int(x) for x in next(iter(obj["prompts"].values()))["raw"])))
        from ..drafting.truthio import raw_to_atom

        oracle = {}
        for rec in obj["prompts"].values():
            atoms = []
            for r in rec["raw"][n]:
                a = raw_to_atom(r)
                if a is None:
                    break
                atoms.append(a)
            oracle[tuple(rec["ids"])] = atoms
    if spec.get("policy", "within") == "gated":
        from . import draft_gated

        stats = draft_gated.GatedStats()
    else:
        stats = core.Stats()
    rec: dict[str, Any] = {"on": False, "raws": []}  # opt-in recording of every emitted raw id (tools/tests)

    def on_round(info: dict) -> None:
        if rec["on"]:
            rec["raws"].extend(info["emitted"])

    gen = make_generate(model, tok, runner, loop, glob, pieces, automaton, engine, spec, ngram=ngram,
                        oracle_truth=oracle, stats=stats, on_round=on_round)
    return {"engine": engine, "generate": gen, "stats": stats, "ngram": ngram, "rec": rec, "oracle": oracle}
