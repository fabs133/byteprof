"""Drafting for OLMo 2 (token model, batch 1, greedy) with the static KV cache and manually captured CUDA graphs.

Same kernels and loop structure as `olmo.make_generate_graph` (eager HF forward captured in a graph, one replay,
one pinned read and one sync per round), but the graph verifies K + 1 tokens: input ids [t0, d_1 .. d_K] at cache
positions p .. p + K, logits for all K + 1 rows, argmax, acceptance (d_j == p_j for the longest prefix, on the
device), and the next round's inputs (t0 <- p_{L+1}, positions p + L + 1 + arange, the mask rows). Rollback of the
KV cache is that position: entries written beyond it are masked and overwritten by the next round.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from . import OptimizeError
from . import draft_core as core
from .olmo import MAX_CACHE_LEN


def build_drafter(spec: dict[str, Any], cfg_dir: str = "results/drafting/corpus", oracle_truth: Any = None) -> Any:
    from pathlib import Path

    from ..drafting.drafters import CombinedDrafter, NullDrafter, OracleDrafter
    from ..drafting.tokens import TokenContextDrafter, TokenNGramDrafter, TokenNGramModel

    kind = spec.get("kind", "none")
    if kind == "none":
        return NullDrafter()
    if kind == "oracle":
        return OracleDrafter()
    ctx = TokenContextDrafter(min_len=spec.get("min_len", 2))
    if kind == "context":
        return ctx
    tokens = np.load(Path(spec.get("corpus_dir", cfg_dir)) / "olmo_tokens.npy")
    model = TokenNGramModel.build(tokens, tuple(spec.get("orders", (1, 2, 3, 4))), bits=22)
    ng = TokenNGramDrafter(model, min_count=spec.get("min_count", 1), max_order=spec.get("max_order", 4))
    if kind == "ngram":
        return ng
    if kind == "combined":
        return CombinedDrafter(ctx, ng)
    raise OptimizeError(f"unknown draft kind {kind!r}")


def load_oracle(spec: dict[str, Any]) -> dict[tuple[int, ...], list[int]]:
    import json
    from pathlib import Path

    obj = json.loads(Path(spec["truth_file"]).read_text(encoding="utf-8"))
    n = str(spec.get("truth_length", max(int(x) for x in next(iter(obj["prompts"].values()))["tokens"])))
    return {tuple(rec["ids"]): rec["tokens"][n] for rec in obj["prompts"].values()}


def make_generate_draft(model: Any, cache: Any, spec: dict[str, Any], stats: core.Stats | None = None) -> tuple[Callable, dict]:
    dev = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    eos_ids = set([eos] if isinstance(eos, int) else (eos or []))
    ks = (0,) + tuple(sorted(k for k in spec.get("ks", (0, 1, 2, 4, 6, 8)) if 0 < k <= spec.get("k_max", 8)))
    k_max = max(ks)
    smax = k_max + 1
    drafter = build_drafter(spec)
    oracle = load_oracle(spec) if spec["kind"] == "oracle" else None
    pinned_d = torch.zeros(smax, dtype=torch.long).pin_memory()
    pinned_o = torch.zeros(smax + 1, dtype=torch.long).pin_memory()
    d_np, o_np = pinned_d.numpy(), pinned_o.numpy()
    done = torch.cuda.Event()
    arange_max = torch.arange(MAX_CACHE_LEN, device=dev)
    arange_s = torch.arange(smax, device=dev)
    st: dict[str, Any] = {"graphs": {}}
    pinned_one = torch.zeros(1, dtype=torch.long).pin_memory()
    pinned_one_np = pinned_one.numpy()

    def build(inputs_len: int, inputs: torch.Tensor) -> None:
        # reference inputs for smax new tokens, built exactly as HF builds them for a decode step
        ids_full = torch.cat([inputs, torch.zeros(1, smax, dtype=inputs.dtype, device=dev)], dim=1)
        mi0 = model.prepare_inputs_for_generation(
            ids_full, past_key_values=cache, attention_mask=torch.ones(1, inputs_len + smax, dtype=inputs.dtype, device=dev),
            cache_position=torch.arange(inputs_len, inputs_len + smax, device=dev), use_cache=True, logits_to_keep=smax)
        ids = mi0["input_ids"].clone()
        cp = mi0["cache_position"].clone()
        pos = mi0["position_ids"].clone() if mi0.get("position_ids") is not None else None
        mask = mi0["attention_mask"].clone()
        is_bool = mask.dtype == torch.bool
        neg = torch.finfo(mask.dtype).min if not is_bool else None
        out_dev = torch.zeros(smax + 1, dtype=torch.long, device=dev)
        st.update(ids=ids, cp=cp, pos=pos, mask=mask, out_dev=out_dev)

        def set_state(first_token_id_tensor: torch.Tensor, base: torch.Tensor) -> None:
            """Inputs for a round starting at cache position `base`: ids[0] = last token, positions, mask rows."""
            if first_token_id_tensor is not None:
                ids[:, :1].copy_(first_token_id_tensor.view(1, 1))
            cp.copy_(base + arange_s)
            if pos is not None:
                pos.copy_(cp.view(1, -1))
            allowed = arange_max[None, :] <= cp[:, None]
            if is_bool:
                mask.copy_(allowed.view_as(mask))
            else:
                mask.copy_(torch.where(allowed, 0.0, neg).to(mask.dtype).view_as(mask))

        st["set_state"] = set_state

        def step(s: int) -> None:
            m = dict(mi0)
            m.update(input_ids=ids[:, :s], cache_position=cp[:s], attention_mask=mask[:, :, :s, :], logits_to_keep=s,
                     past_key_values=cache, use_cache=True)
            if pos is not None:
                m["position_ids"] = pos[:, :s]
            out = model(**m, return_dict=True)
            tok = torch.argmax(out.logits[0].to(torch.float32), dim=-1)  # (s,)
            length = core.accepted_length_t(ids[0], tok, s)
            out_dev[:s].copy_(tok)
            out_dev[smax].copy_(length)
            set_state(tok.index_select(0, length.view(1)), cp[0] + length + 1)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(2):  # compile / initialize kernels before capture (only writes positions >= the prompt)
                for k in ks:
                    step(k + 1)
                set_state(None, torch.tensor(inputs_len, device=dev))
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for k in ks:
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g):
                step(k + 1)
            st["graphs"][k] = g
        torch.cuda.synchronize()

    @torch.no_grad()
    def generate(inputs: torch.Tensor, generation_config: Any = None, **kwargs: Any) -> torch.Tensor:
        if inputs.shape[0] != 1 or kwargs.get("do_sample"):
            raise OptimizeError("the drafting loop supports batch size 1 and greedy decoding only")
        max_new = kwargs["max_new_tokens"]
        n_prompt = inputs.shape[1]
        if n_prompt + max_new + smax > MAX_CACHE_LEN:
            raise OptimizeError(f"{n_prompt}+{max_new} tokens exceed the static cache ({MAX_CACHE_LEN})")
        cache.reset()
        mi = model.prepare_inputs_for_generation(
            inputs, past_key_values=cache, attention_mask=torch.ones_like(inputs),
            cache_position=torch.arange(n_prompt, device=dev), use_cache=True, logits_to_keep=1)
        out = model(**mi, return_dict=True)
        t0 = torch.argmax(out.logits[:, -1, :].to(torch.float32), dim=-1)
        pinned_one.copy_(t0, non_blocking=True)
        done.record()
        done.synchronize()
        new = [0] * (max_new + smax)
        new[0] = int(pinned_one_np[0])
        n = 1
        if new[0] not in eos_ids and n < max_new:
            if "ids" not in st:
                build(n_prompt, inputs)  # first decode: warm up + capture (writes cache positions >= n_prompt only)
            st["set_state"](t0, torch.tensor(n_prompt, device=dev))
            ctx = inputs[0].tolist() + [new[0]]
            truth = None
            if oracle is not None:
                truth = oracle.get(tuple(inputs[0].tolist()))
                if truth is None:
                    raise OptimizeError("oracle drafter: no recorded ground truth for this prompt")
            drafter.reset(ctx, truth)
            ids_dev = st["ids"]
            out_dev = st["out_dev"]
            graphs = st["graphs"]
            last = new[0]
            while True:
                draft = drafter.propose(k_max) if (k_max > 0 and len(ks) > 1) else []
                k = core.choose_k(len(draft), ks, k_max)
                if k:
                    d_np[:k] = draft[:k]
                    ids_dev[:, 1:1 + k].copy_(pinned_d[:k].view(1, k), non_blocking=True)
                graphs[k].replay()
                pinned_o.copy_(out_dev, non_blocking=True)
                done.record()
                done.synchronize()  # the single sync of this round
                toks = o_np[:k + 1].tolist()
                length = int(o_np[smax])
                if length != core.accepted_length([last] + draft[:k], toks, k):
                    raise RuntimeError("host and device disagree on the accepted length")
                emitted = toks[:length + 1]
                stop = False
                n_before = n
                for r in emitted:
                    new[n] = r
                    n += 1
                    if r in eos_ids or n >= max_new:
                        stop = True
                        break
                if stats is not None:
                    stats.rounds += 1
                    stats.emitted += n - n_before
                    stats.by_k[k] = stats.by_k.get(k, 0) + 1
                    stats.l_hist[length] = stats.l_hist.get(length, 0) + 1
                if stop:
                    break
                drafter.push(emitted)
                last = emitted[-1]
        gen = torch.tensor(new[:n], dtype=torch.long, device=dev)
        return torch.cat([inputs[0], gen]).unsqueeze(0)

    generate.drafter = drafter  # type: ignore[attr-defined]
    return generate, st
