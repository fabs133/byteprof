"""OLMo 2 (token model): static KV cache + torch.compile of the decode forward.

`decode_loop = "hf"` uses HF generate() with the static cache (its auto-compile of the decode
forward). `decode_loop = "minimal"` replaces generate() with a loop that calls the very same
compiled forward but does the rest with one device-to-host read per token: argmax on the GPU,
inputs for the next step updated in place, preallocated output list.
"""

from __future__ import annotations

from typing import Any

from ..config import ProfileConfig
from . import Handle, OptimizeError

# One cache shape for every prompt, so compiled graphs are reused across prompts.
MAX_CACHE_LEN = 1024


class OlmoHandle(Handle):
    def __init__(self, cache: Any, compile_config: Any, backend: str, minimal: bool = False,
                 closers: list[Any] | None = None, graph: bool = False):
        self.cache, self.compile_config, self.backend, self.minimal = cache, compile_config, backend, minimal
        self.graph = graph
        self._closers = closers or []
        self.extra = ""  # appended to describe(); set by the opt-in drafting loop

    def generate_kwargs(self) -> dict[str, Any]:
        import torch

        if self.minimal:  # the minimal generate() resets the cache and compiles by itself
            return {}
        with torch.inference_mode():  # cache tensors are created lazily inside generate()'s inference mode
            self.cache.reset()
        return {"past_key_values": self.cache, "compile_config": self.compile_config}

    def describe(self) -> str:
        if self.graph:
            return (self.extra + f"static KV cache (max_cache_len={MAX_CACHE_LEN}) + manual CUDA graph of the eager decode forward "
                    "incl. argmax and next-step inputs (no torch.compile); prefill eager; decode loop: one replay, "
                    "one pinned read, one sync per token")
        loop = ("byteprof minimal loop (one read per token, same compiled forward)" if self.minimal
                else "HF generate() loop")
        return (f"static KV cache (max_cache_len={MAX_CACHE_LEN}) + compiled decode forward "
                f"(torch.compile backend={self.backend!r}, mode='reduce-overhead', fullgraph=True, dynamic=False); "
                f"prefill stays eager; decode loop: {loop}")

    def close(self) -> None:
        for fn in self._closers:
            fn()


def make_generate(model: Any, cache: Any, cc: Any):
    """Batch-1 greedy generate() replacement with one device-to-host read per token."""
    import torch

    dev = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    eos_ids = set([eos] if isinstance(eos, int) else (eos or []))
    pinned = torch.zeros(1, dtype=torch.long).pin_memory()
    pinned_np = pinned.numpy()
    done = torch.cuda.Event()
    arange = torch.arange(MAX_CACHE_LEN, device=dev)

    @torch.no_grad()
    def generate(inputs: torch.Tensor, generation_config: Any = None, **kwargs: Any) -> torch.Tensor:
        if inputs.shape[0] != 1 or kwargs.get("do_sample"):
            raise OptimizeError("minimal loop supports batch size 1 and greedy decoding only")
        max_new = kwargs["max_new_tokens"]
        L = inputs.shape[1]
        if L + max_new > MAX_CACHE_LEN:
            raise OptimizeError(f"{L}+{max_new} tokens exceed the static cache ({MAX_CACHE_LEN})")
        cache.reset()
        # prefill: exactly the inputs HF generate() builds for it
        mi = model.prepare_inputs_for_generation(
            inputs, past_key_values=cache, attention_mask=torch.ones_like(inputs),
            cache_position=torch.arange(L, device=dev), use_cache=True, logits_to_keep=1)
        out = model(**mi, return_dict=True)
        tok = torch.argmax(out.logits[:, -1, :].to(torch.float32), dim=-1)
        pinned.copy_(tok, non_blocking=True)
        done.record()
        done.synchronize()
        new = [0] * max_new
        new[0] = int(pinned_np[0])
        n = 1
        if new[0] not in eos_ids and n < max_new:
            # decode inputs as HF builds them for step 1; then updated in place every step
            mi = model.prepare_inputs_for_generation(
                torch.cat([inputs, tok.view(1, 1)], dim=1), past_key_values=cache,
                attention_mask=torch.ones(1, L + 1, dtype=inputs.dtype, device=dev),
                cache_position=torch.tensor([L], device=dev), use_cache=True, logits_to_keep=1)
            fwd = model.get_compiled_call(cc)
            ids, cp, pos_ids, mask = mi["input_ids"], mi["cache_position"], mi.get("position_ids"), mi["attention_mask"]
            is_bool = mask.dtype == torch.bool
            neg = torch.finfo(mask.dtype).min if not is_bool else None
            while True:
                ids.copy_(tok.view(1, 1))
                out = fwd(**mi, return_dict=True)
                tok = torch.argmax(out.logits[:, -1, :].to(torch.float32), dim=-1)
                pinned.copy_(tok, non_blocking=True)
                done.record()
                done.synchronize()  # the single sync of this token
                v = int(pinned_np[0])
                new[n] = v
                n += 1
                if v in eos_ids or n >= max_new:
                    break
                cp.add_(1)
                if pos_ids is not None:
                    pos_ids.add_(1)
                allowed = arange <= cp
                if is_bool:
                    mask.copy_(allowed.view_as(mask))
                else:
                    mask.copy_(torch.where(allowed, 0.0, neg).to(mask.dtype).view_as(mask))
        gen = torch.tensor(new[:n], dtype=torch.long, device=dev)
        return torch.cat([inputs[0], gen]).unsqueeze(0)

    return generate


def make_generate_graph(model: Any, cache: Any):
    """Like make_generate, but the decode step is one manually captured CUDA graph of the *eager*
    forward (no torch.compile): forward, argmax, next-step inputs (token, cache position, position
    ids, mask) are all updated inside the graph. Per token: one replay, one device-to-host copy of
    the token into pinned memory, one event synchronize. This is the same kernel path as Bolmo's
    global graph (eager HF layers captured), so quantized kernels are the same ones."""
    import torch

    dev = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    eos_ids = set([eos] if isinstance(eos, int) else (eos or []))
    pinned = torch.zeros(1, dtype=torch.long).pin_memory()
    pinned_np = pinned.numpy()
    done = torch.cuda.Event()
    arange = torch.arange(MAX_CACHE_LEN, device=dev)
    st: dict[str, Any] = {}

    def build(mi: dict[str, Any]) -> None:
        keys = [k for k in ("input_ids", "cache_position", "position_ids", "attention_mask") if mi.get(k) is not None]
        m = dict(mi)
        for k in keys:
            m[k] = mi[k].clone()
        st["mi"], st["keys"] = m, keys
        st["tok"] = torch.zeros(1, dtype=torch.long, device=dev)
        mask = m["attention_mask"]
        is_bool = mask.dtype == torch.bool
        neg = torch.finfo(mask.dtype).min if not is_bool else None

        def step() -> None:
            out = model(**m, return_dict=True)
            tok = torch.argmax(out.logits[:, -1, :].to(torch.float32), dim=-1)
            st["tok"].copy_(tok)
            m["input_ids"].copy_(tok.view(1, 1))
            m["cache_position"].add_(1)
            if m.get("position_ids") is not None:
                m["position_ids"].add_(1)
            allowed = arange <= m["cache_position"]
            if is_bool:
                mask.copy_(allowed.view_as(mask))
            else:
                mask.copy_(torch.where(allowed, 0.0, neg).to(mask.dtype).view_as(mask))

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):  # initialize kernels before capture (mutates the static inputs)
                step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            step()
        st["graph"] = g

    @torch.no_grad()
    def generate(inputs: torch.Tensor, generation_config: Any = None, **kwargs: Any) -> torch.Tensor:
        if inputs.shape[0] != 1 or kwargs.get("do_sample"):
            raise OptimizeError("minimal loop supports batch size 1 and greedy decoding only")
        max_new = kwargs["max_new_tokens"]
        L = inputs.shape[1]
        if L + max_new > MAX_CACHE_LEN:
            raise OptimizeError(f"{L}+{max_new} tokens exceed the static cache ({MAX_CACHE_LEN})")
        cache.reset()
        mi = model.prepare_inputs_for_generation(
            inputs, past_key_values=cache, attention_mask=torch.ones_like(inputs),
            cache_position=torch.arange(L, device=dev), use_cache=True, logits_to_keep=1)
        out = model(**mi, return_dict=True)
        tok = torch.argmax(out.logits[:, -1, :].to(torch.float32), dim=-1)
        pinned.copy_(tok, non_blocking=True)
        done.record()
        done.synchronize()
        new = [0] * max_new
        new[0] = int(pinned_np[0])
        n = 1
        if new[0] not in eos_ids and n < max_new:
            mi = model.prepare_inputs_for_generation(
                torch.cat([inputs, tok.view(1, 1)], dim=1), past_key_values=cache,
                attention_mask=torch.ones(1, L + 1, dtype=inputs.dtype, device=dev),
                cache_position=torch.tensor([L], device=dev), use_cache=True, logits_to_keep=1)
            if "graph" not in st:
                build(mi)  # first decode: warm up + capture (warmup only writes cache positions >= L)
            for k in st["keys"]:
                st["mi"][k].copy_(mi[k])
            g, tok_buf = st["graph"], st["tok"]
            while True:
                g.replay()
                pinned.copy_(tok_buf, non_blocking=True)
                done.record()
                done.synchronize()  # the single sync of this token
                v = int(pinned_np[0])
                new[n] = v
                n += 1
                if v in eos_ids or n >= max_new:
                    break
        gen = torch.tensor(new[:n], dtype=torch.long, device=dev)
        return torch.cat([inputs[0], gen]).unsqueeze(0)

    return generate


def apply(model: Any, tok: Any, cfg: ProfileConfig) -> OlmoHandle:
    import torch
    from transformers import CompileConfig, StaticCache

    if not cfg.device.startswith("cuda"):
        raise OptimizeError("compile variant needs a CUDA device")
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAX_CACHE_LEN,
                        device=cfg.device, dtype=getattr(torch, cfg.dtype))
    if cfg.optimize == "cuda_graph":
        if cfg.decode_loop != "minimal":
            raise OptimizeError("OLMo optimize = 'cuda_graph' needs decode_loop = 'minimal'")
        if cfg.draft.get("kind", "none") != "none":  # opt-in drafting loop (TASK_drafting.md); never reached by default
            from . import draft_core, olmo_draft

            stats = draft_core.Stats()
            model.generate, state = olmo_draft.make_generate_draft(model, cache, dict(cfg.draft), stats)
            handle = OlmoHandle(cache, None, "none", True, [lambda: delattr(model, "generate")], graph=True)
            handle.stats, handle.draft_state = stats, state
            handle.extra = f"drafting loop {dict(cfg.draft)}: "
            return handle
        model.generate = make_generate_graph(model, cache)
        return OlmoHandle(cache, None, "none", True, [lambda: delattr(model, "generate")], graph=True)
    cc = CompileConfig(backend=cfg.compile_backend, mode="reduce-overhead", fullgraph=True, dynamic=False)
    closers: list[Any] = []
    minimal = cfg.decode_loop == "minimal"
    if minimal:
        model.generate = make_generate(model, cache, cc)
        closers.append(lambda: delattr(model, "generate"))
    return OlmoHandle(cache, cc, cfg.compile_backend, minimal, closers)
