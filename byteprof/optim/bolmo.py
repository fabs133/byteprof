"""Bolmo: optimize the per-byte local loop (local encoder step + local decoder step).

See notes/bolmo_decode_loop.md. The decode step is split into three pieces:

  A  encoder step   (captured / compiled)   byte id, subword id -> h_byte, patch embedding
  B  global model   (eager, boundary steps only, dynamic KV cache)
  C  decoder step   (captured / compiled)   h_byte + last_value -> decoder output

Both A and C are *functional over static buffers*: they read input buffers and write output
and recurrent-state buffers in place, so the same function can be captured by
torch.cuda.CUDAGraph or compiled with mode="reduce-overhead". The boundary branch, the
global-model call and the whole generate() loop stay in Python. Only batch size 1 decode
steps take this path; prefill and anything else run the original code.

Implemented by replacing `forward` on the local_encoder / local_decoder *instances* (Bolmo's
BolmoModel calls local_encoder via __call__ and local_decoder.forward directly, so both are
reached through the instance attribute).
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn

from ..config import ProfileConfig
from . import Handle, OptimizeError


def _layer_step(layer: Any, h: torch.Tensor, state: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """BolmoLocalLayer.forward for one decode step, state updated in place.

    Same ops as the eager path (xlstm._original_forward, then the in-place copy that
    MaskState.selective_put performs at batch 1), so results are bit-identical.
    """
    xl = layer.xlstm
    xl.mlstm_backend.config.mode = "inference"
    y, new_state = xl._original_forward(layer.pre_xlstm_layernorm(h), state=state)
    for dst, src in zip(state, new_state):
        dst.copy_(src)
    h = h + y
    return h + layer.mlp(layer.pre_feedforward_layernorm(h))


class LocalLoop(nn.Module):
    """Static buffers + the two step functions."""

    def __init__(self, model: Any, dtype: torch.dtype, device: str):
        super().__init__()
        self.enc = model.model.local_encoder
        self.dec = model.model.local_decoder
        d = self.enc.hidden_size
        self.d = d

        def states(layers: Any) -> list[tuple[torch.Tensor, ...]]:
            out = []
            for layer in layers:
                xl = layer.xlstm
                nh = xl.config.num_heads
                out.append((
                    torch.zeros(1, nh, xl.qk_dim // nh, xl.v_dim // nh, device=device, dtype=torch.float32),
                    torch.zeros(1, nh, xl.qk_dim // nh, device=device, dtype=torch.float32),
                    torch.zeros(1, nh, 1, device=device, dtype=torch.float32),
                ))
            return out

        self.enc_states = states(self.enc.layers)
        self.dec_states = states(self.dec.layers)
        for i, st in enumerate(self.enc_states + self.dec_states):
            for j, t in enumerate(st):
                self.register_buffer(f"state_{i}_{j}", t, persistent=False)
        self.register_buffer("byte_ids", torch.zeros(1, 1, dtype=torch.long, device=device), persistent=False)
        self.register_buffer("exp_ids", torch.zeros(1, 1, dtype=torch.long, device=device), persistent=False)
        self.register_buffer("h_byte", torch.zeros(1, 1, d, dtype=dtype, device=device), persistent=False)
        self.register_buffer("patch_out", torch.zeros(1, 1, d, dtype=dtype, device=device), persistent=False)
        self.register_buffer("dec_out", torch.zeros(1, 1, d, dtype=dtype, device=device), persistent=False)
        self.register_buffer("last_value", torch.zeros(1, d, dtype=dtype, device=device), persistent=False)

    # A: encoder step -------------------------------------------------------------------
    def enc_step(self) -> None:
        enc = self.enc
        h = enc._embed(self.byte_ids, self.exp_ids)
        for layer, st in zip(enc.layers, self.enc_states):
            h = _layer_step(layer, h, st)
        h = enc.post_last_block_norm(h)
        self.h_byte.copy_(h)
        self.patch_out.copy_(enc.out_projection(h))

    # C: decoder step -------------------------------------------------------------------
    def dec_step(self) -> None:
        dec = self.dec
        h = dec.in_projection(self.h_byte) + self.last_value.unsqueeze(1)
        for layer, st in zip(dec.layers, self.dec_states):
            h = _layer_step(layer, h, st)
        self.dec_out.copy_(h)


class GlobalStep(nn.Module):
    """The 16 global layers for one patch, on a static KV cache of fixed length (stage 3b)."""

    MAX_PATCHES = 512

    def __init__(self, model: Any, loop: LocalLoop, dtype: torch.dtype, device: str):
        super().__init__()
        from transformers import StaticCache

        self.m = model.model
        self.loop = loop
        self.cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=self.MAX_PATCHES,
                                 device=device, dtype=dtype)
        self.register_buffer("pos", torch.zeros(1, dtype=torch.long, device=device), persistent=False)
        self.register_buffer("kv_index", torch.arange(self.MAX_PATCHES, device=device), persistent=False)
        self.register_buffer("glob_out", torch.zeros(1, 1, loop.d, dtype=dtype, device=device), persistent=False)

    def step(self) -> None:
        m = self.m
        h = self.loop.patch_out
        position_ids = self.pos.unsqueeze(0)
        pe = m.rotary_embs["full_attention"](self.loop.h_byte, position_ids)
        mask = (self.kv_index <= self.pos).view(1, 1, 1, -1)
        for layer in m.layers:
            h = layer(h, attention_mask=mask, position_ids=position_ids, past_key_values=self.cache,
                      cache_position=self.pos, position_embeddings=pe)
        self.glob_out.copy_(h)
        self.pos.add_(1)

    def load_prefill(self, dyn: Any) -> None:
        n = dyn.get_seq_length()
        if n >= self.MAX_PATCHES:
            raise OptimizeError(f"{n} prefill patches exceed the static global cache ({self.MAX_PATCHES})")
        for sl, dl in zip(self.cache.layers, dyn.layers):
            sl.keys[:, :, :n].copy_(dl.keys)
            sl.values[:, :, :n].copy_(dl.values)
        self.pos.fill_(n)


class Runner:
    """Calls the step functions through a CUDA graph or a compiled callable."""

    def __init__(self, loop: LocalLoop, mode: str, backend: str, glob: GlobalStep | None = None):
        self.loop, self.mode, self.backend, self.glob = loop, mode, backend, glob
        self.graphs: dict[str, torch.cuda.CUDAGraph] = {}
        self.capture_seconds = 0.0
        self.fns: dict[str, Any] = {"enc": loop.enc_step, "dec": loop.dec_step}
        if mode == "compile":
            self.fns = {k: torch.compile(v, backend=backend, mode="reduce-overhead", fullgraph=False, dynamic=False)
                        for k, v in self.fns.items()}

    def warmup_and_capture(self) -> None:
        import time

        t0 = time.perf_counter()
        torch.cuda.synchronize()
        if self.mode == "cuda_graph":
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.inference_mode():
                for _ in range(3):  # compile the Triton kernels before capture
                    self.loop.enc_step()
                    self.loop.dec_step()
                    if self.glob is not None:
                        self.glob.step()
                if self.glob is not None:
                    self.glob.pos.zero_()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            fns = [("enc", self.loop.enc_step), ("dec", self.loop.dec_step)]
            if self.glob is not None:
                fns.append(("glob", self.glob.step))
            for name, fn in fns:
                g = torch.cuda.CUDAGraph()
                with torch.inference_mode(), torch.cuda.graph(g):
                    fn()
                self.graphs[name] = g
        else:
            with torch.inference_mode():
                for _ in range(4):
                    self.fns["enc"]()
                    self.fns["dec"]()
        torch.cuda.synchronize()
        self.capture_seconds = time.perf_counter() - t0

    def run(self, name: str) -> None:
        if self.mode == "cuda_graph":
            self.graphs[name].replay()
        else:
            self.fns[name]()


class Instr:
    """Opt-in instrumentation of the decode phase of one generation.

    "timing": every graph replay is bracketed by two CUDA events (GPU time, resolved after the
    generation, so no extra sync) and by perf_counter (host time spent launching). "syncs": counts
    host syncs after the first decode step, with torch.cuda.set_sync_debug_mode("warn") (total)
    and with counting wrappers on item/cpu/tolist/__bool__/synchronize (categories).
    """

    def __init__(self, runner: "Runner", kind: str):
        self.runner, self.kind = runner, kind
        self._orig_run = runner.run
        runner.run = self._run  # type: ignore[method-assign]
        self.reset()

    def reset(self) -> None:
        self.t_start = time.perf_counter()
        self.t_first = None
        self.events: list[tuple[str, Any, Any]] = []
        self.host: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.cats: dict[str, int] = {}
        self._ctx = None
        self._warn = None

    def _run(self, name: str) -> None:
        if self.t_first is None and name.startswith("enc"):
            self.t_first = time.perf_counter()
            if self.kind == "syncs":
                self._start_sync_count()
        if self.kind == "timing":
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            a.record()
            self._orig_run(name)
            b.record()
            self.host[name] = self.host.get(name, 0.0) + (time.perf_counter() - t0) * 1000
            self.events.append((name, a, b))
        else:
            self._orig_run(name)
        self.calls[name] = self.calls.get(name, 0) + 1

    # -- sync counting -----------------------------------------------------------------
    def _start_sync_count(self) -> None:
        import warnings

        self._ctx = warnings.catch_warnings(record=True)
        self._warn = self._ctx.__enter__()
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        cats = self.cats
        self._patched = []
        for owner, attr in ((torch.Tensor, "item"), (torch.Tensor, "cpu"), (torch.Tensor, "tolist"),
                            (torch.Tensor, "__bool__"), (torch.cuda, "synchronize"), (torch.cuda.Event, "synchronize")):
            orig = getattr(owner, attr)

            def wrapper(*a, _o=orig, _k=attr, **kw):
                cats[_k] = cats.get(_k, 0) + 1
                return _o(*a, **kw)

            setattr(owner, attr, wrapper)
            self._patched.append((owner, attr, orig))

    def _stop_sync_count(self) -> int:
        torch.cuda.set_sync_debug_mode(0)
        for owner, attr, orig in self._patched:
            setattr(owner, attr, orig)
        n = sum(1 for w in self._warn if "synchroniz" in str(w.message))
        self._ctx.__exit__(None, None, None)
        return n

    def finish(self) -> dict[str, Any]:
        t_end = time.perf_counter()
        out: dict[str, Any] = {}
        steps = sum(v for k, v in self.calls.items() if k.startswith("enc"))
        t_first = self.t_first if self.t_first is not None else t_end
        out["prefill_ms"] = (t_first - self.t_start) * 1000
        out["decode_wall_ms"] = (t_end - t_first) * 1000
        out["decode_steps"] = steps
        if self.kind == "timing":
            torch.cuda.synchronize()
            gpu: dict[str, float] = {}
            for name, a, b in self.events:
                gpu[name] = gpu.get(name, 0.0) + a.elapsed_time(b)
            for grp in ("enc", "dec", "glob"):  # enc_m / dec_n / dec_p fold into their group
                out[f"gpu_ms_{grp}"] = sum(v for k, v in gpu.items() if k.startswith(grp))
                out[f"host_ms_{grp}"] = sum(v for k, v in self.host.items() if k.startswith(grp))
            out["gpu_ms_replays"] = sum(gpu.values())
            out["outside_replays_ms"] = out["decode_wall_ms"] - out["gpu_ms_replays"]
            out["glob_calls"] = self.calls.get("glob", 0)
            out["host_ms_total_in_replays"] = sum(self.host.values())
        else:
            if self._ctx is not None:
                out["syncs_total"] = self._stop_sync_count()
                out.update({f"calls_{k.strip('_')}": v for k, v in self.cats.items()})
                out["syncs_per_step"] = out["syncs_total"] / max(steps, 1)
        return out


class BolmoHandle(Handle):
    def __init__(self, runner: Runner, restore: list[tuple[Any, Any]], instr: "Instr | None" = None,
                 closers: list[Any] | None = None):
        self.runner, self._restore, self.instr = runner, restore, instr
        self._closers = closers or []

    def begin(self) -> None:
        if self.instr is not None:
            self.instr.reset()

    def end(self) -> dict[str, Any] | None:
        return self.instr.finish() if self.instr is not None else None

    def describe(self) -> str:
        r = self.runner
        what = (f"manual torch.cuda.CUDAGraph capture of the local encoder step, the local decoder step"
                + (" and the 16 global layers (static KV cache)" if r.glob is not None else "")
                if r.mode == "cuda_graph" else
                f"torch.compile(backend={r.backend!r}, mode='reduce-overhead') of the local encoder step and decoder step")
        return (f"{what}; boundary branch, "
                f"{'' if r.glob is not None else 'global model, '}final norm, lm_head and generate() stay eager; "
                f"warmup+capture {r.capture_seconds:.1f} s"
                + ("; decode loop: minimal (one sync per byte, automaton subword ids)" if self._closers else ""))

    def close(self) -> None:
        for fn in self._closers:
            fn()
        for mod, fwd in self._restore:
            mod.forward = fwd


def _sync_states(runner: Runner, src_layers: list[dict], dst: list[tuple[torch.Tensor, ...]]) -> None:
    for layer_state, bufs in zip(src_layers, dst):
        state = layer_state["xlstm"].get("state")
        if state is None:
            raise OptimizeError("recurrent state missing after prefill")
        for b, s in zip(bufs, state):
            b.copy_(s)


def apply(model: Any, tok: Any, cfg: ProfileConfig) -> BolmoHandle:
    if not cfg.device.startswith("cuda"):
        raise OptimizeError("Bolmo local-loop optimization needs a CUDA device")
    dtype = getattr(torch, cfg.dtype)
    loop = LocalLoop(model, dtype, cfg.device)
    mode = "cuda_graph" if cfg.optimize == "cuda_graph" else "compile"
    glob = GlobalStep(model, loop, dtype, cfg.device) if cfg.optimize_global else None
    runner = Runner(loop, mode, cfg.compile_backend, glob)
    runner.warmup_and_capture()

    enc, dec = loop.enc, loop.dec
    orig_enc_forward, orig_dec_forward = enc.forward, dec.forward
    seen = {"enc": None, "dec": None}
    empty = torch.empty(0, 1, loop.d, dtype=dtype, device=cfg.device)

    def enc_forward(input_ids, true_boundary_mask=None, boundary_state=None, pad_state=None,
                    expanded_input_ids=None, sequence_start_indices=None):
        decode = enc.has_cache and enc.cache_seqlens > 0
        if not decode:
            return orig_enc_forward(input_ids, true_boundary_mask, boundary_state, pad_state,
                                    expanded_input_ids, sequence_start_indices)
        if input_ids.shape != (1, 1) or pad_state.all() or true_boundary_mask is not None:
            raise OptimizeError("unsupported decode step for the optimized local encoder (batch>1 or idle row)")
        if seen["enc"] is not enc.layer_states:  # new generation: load the prefill state
            _sync_states(runner, [ls for ls in enc.layer_states], loop.enc_states)
            seen["enc"] = enc.layer_states
        loop.byte_ids.copy_(input_ids)
        loop.exp_ids.copy_(expanded_input_ids)
        runner.run("enc")
        enc.last_h.copy_(loop.h_byte[:, -1])
        enc.cache_seqlens += 1
        patch = loop.patch_out if boundary_state.all() else empty
        return loop.h_byte, patch, None, None

    def dec_forward(embeds, patch_embeds, boundary_state, boundary_mask, sequence_start_indices=None):
        decode = dec.has_cache and dec.cache_seqlens > 0
        if not decode:
            return orig_dec_forward(embeds, patch_embeds, boundary_state, boundary_mask, sequence_start_indices)
        if embeds.data_ptr() != loop.h_byte.data_ptr():
            raise OptimizeError("decoder input is not the encoder's static output buffer")
        if seen["dec"] is not dec.layer_states:
            _sync_states(runner, [ls for ls in dec.layer_states], loop.dec_states)
            seen["dec"] = dec.layer_states
            loop.last_value.copy_(dec.last_value)
        if patch_embeds.numel() > 0:  # boundary step: new value from the global model
            h_patch = dec.initial_norm(patch_embeds)[:, -1:, :]
            loop.last_value.copy_(h_patch[:, -1])
            dec.last_value.copy_(h_patch[:, -1])
        runner.run("dec")
        dec.cache_seqlens += 1
        return loop.dec_out

    enc.forward = enc_forward
    dec.forward = dec_forward
    restore = [(enc, orig_enc_forward), (dec, orig_dec_forward)]

    if glob is not None:
        from transformers.modeling_outputs import BaseModelOutputWithPast

        bm = model.model
        orig_model_forward = bm.forward
        seen_pkv: dict[str, Any] = {"p": None}

        def model_forward(input_ids, expanded_input_ids=None, attention_mask=None, position_ids=None,
                          past_key_values=None, cache_position=None, use_cache=None, boundary_mask=None,
                          boundary_state=None, pad_state=None, sequence_start_indices=None, **kwargs):
            if not (boundary_mask is None and enc.has_cache and enc.cache_seqlens > 0
                    and input_ids.shape == (1, 1) and past_key_values is not None):
                return orig_model_forward(
                    input_ids=input_ids, expanded_input_ids=expanded_input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values, cache_position=cache_position,
                    use_cache=use_cache, boundary_mask=boundary_mask, boundary_state=boundary_state,
                    pad_state=pad_state, sequence_start_indices=sequence_start_indices, **kwargs)
            h_byte, h_patch, _, _ = enc(input_ids=input_ids, expanded_input_ids=expanded_input_ids,
                                        boundary_state=boundary_state, pad_state=pad_state)
            if seen_pkv["p"] is not past_key_values:  # new generation: copy the prefill KV cache
                glob.load_prefill(past_key_values)
                seen_pkv["p"] = past_key_values
            if h_patch.numel() > 0:
                runner.run("glob")
                h_patch = glob.glob_out
            h_out = dec.forward(embeds=h_byte, patch_embeds=h_patch, boundary_mask=None,
                                boundary_state=boundary_state, sequence_start_indices=sequence_start_indices)
            return BaseModelOutputWithPast(last_hidden_state=bm.norm(h_out), past_key_values=past_key_values)

        bm.forward = model_forward
        restore.append((bm, orig_model_forward))
    closers: list[Any] = []
    pieces = automaton = None
    if cfg.decode_loop == "minimal":
        if glob is None or runner.mode != "cuda_graph":
            raise OptimizeError("decode_loop = 'minimal' needs optimize = 'cuda_graph' and optimize_global = true")
        from . import bolmo_loop
        from .subword_automaton import SubwordAutomaton

        automaton = SubwordAutomaton.from_tokenizer(model.model.tokenizer)
        pieces = bolmo_loop.MinimalPieces(model, loop, glob, cfg.device)
        bolmo_loop.capture_pieces(runner, pieces)
        model.generate = bolmo_loop.make_generate(model, tok, runner, loop, glob, pieces, automaton)
        closers.append(lambda: delattr(model, "generate"))
        if cfg.draft.get("kind", "none") != "none":  # opt-in drafting loop (TASK_drafting.md); never reached by default
            from . import bolmo_draft

            drafting = bolmo_draft.install(model, tok, cfg, runner, loop, glob, pieces, automaton)
            model.generate = drafting["generate"]
    instr = Instr(runner, cfg.instrument) if cfg.instrument != "none" else None
    handle = BolmoHandle(runner, restore, instr, closers)
    handle.loop, handle.glob, handle.pieces, handle.automaton = loop, glob, pieces, automaton  # for tools/debug
    handle.drafting = drafting if cfg.draft.get("kind", "none") != "none" and cfg.decode_loop == "minimal" else None
    return handle
