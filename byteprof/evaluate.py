"""Quality evaluation of a (possibly quantized) model: bits per byte and CUTE accuracy.

`python -m byteprof eval --config <toml> --task bpb|cute [--limit N] [--output-dir DIR]` writes
DIR/eval.json. Both tasks run on the model exactly as `load_model` returns it (so after weight
quantization if the config has a `[quantize]` table). CUTE generation uses the config's
`optimize` / `decode_loop` settings, i.e. the same graphs and loop that the speed runs use.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ProfileConfig
from .stats import bootstrap_ci, ratio_of_sums, stratified_sample, superset_sample

log = logging.getLogger("byteprof")

WIKITEXT_REPO, WIKITEXT_REV = "Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
CUTE_REPO, CUTE_REV = "leukas/cute", "34e3d23fe05fdd43bae9ea609cfd9b33f071cd65"
CUTE_TASKS = ("contains_char", "contains_word", "del_char", "del_word", "ins_char", "ins_word", "orth", "sem",
              "spell", "spell_inverse", "sub_char", "sub_word", "swap_char", "swap_word")
BPB_BYTES = 200_000
BPB_CHUNK = 1024
CUTE_ITEMS = 500
CUTE_SEED = 20261026
CUTE_SUFFIX = ' Answer: "'


def _hub_file(repo: str, filename: str, revision: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


def load_wikitext_bytes(n_bytes: int = BPB_BYTES) -> bytes:
    import pyarrow.parquet as pq

    table = pq.read_table(_hub_file(WIKITEXT_REPO, WIKITEXT_FILE, WIKITEXT_REV))
    return "".join(table.column("text").to_pylist()).encode("utf-8")[:n_bytes]


def chunk_bytes(data: bytes, size: int = BPB_CHUNK) -> list[bytes]:
    """Fixed-length byte chunks (the last may be shorter); identical for every variant."""
    return [data[i:i + size] for i in range(0, len(data), size)]


def load_cute() -> dict[str, list[dict[str, str]]]:
    import pyarrow.parquet as pq

    return {t: pq.read_table(_hub_file(CUTE_REPO, f"data/{t}-00000-of-00001.parquet", CUTE_REV)).to_pylist()
            for t in CUTE_TASKS}


def cute_extract(generated: str) -> str:
    """The model's answer: text up to the first closing quote, whitespace-stripped."""
    return generated.split('"', 1)[0].strip()


def cute_correct(generated: str, gold: str) -> bool:
    return cute_extract(generated) == gold.strip()


# ---------------------------------------------------------------------------
# bits per byte
# ---------------------------------------------------------------------------


def _bolmo_chunk(model: Any, tok: Any, chunk: bytes, ms: Any) -> dict[str, float]:
    """Teacher-forced NLL of one chunk. Bolmo's lm_head predicts fused (byte, patch-boundary) tokens
    (class = byte id + 260 if the byte ends a patch). Boundary flags come from the model's own
    boundary predictor over the full chunk, like the prefill of generate(). Primary number: NLL of
    the fused tokens (an upper bound on the byte NLL, since the flags are coded too); secondary: the
    marginal over the flag, -log(p(b,0) + p(b,1))."""
    import torch

    dev = next(model.parameters()).device
    ids = torch.tensor([[tok.bos_token_id] + [b + tok.offset for b in chunk]], device=dev)
    exp = torch.tensor([tok.expand_byte_ids(ids[0].tolist())], device=dev)
    start = torch.zeros(1, dtype=torch.long, device=dev)
    bm_model = model.model
    bm_model.local_encoder.free_inference_cache()
    bm_model.local_decoder.free_inference_cache()
    with torch.inference_mode():
        bm = bm_model.prefill_boundary_prediction_forward(ids, expanded_input_ids=exp, sequence_start_indices=start)
        out = model(ids, expanded_input_ids=exp, boundary_mask=bm,
                    boundary_state=ms(bm[:, -1].clone()), pad_state=ms(torch.zeros(1, dtype=torch.bool, device=dev)),
                    sequence_start_indices=start, use_cache=False)
        logp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)  # position i predicts token i + 1
        plain = ids[0, 1:]
        flag = bm[0, 1:].long()
        fused = logp.gather(1, (plain + 260 * flag)[:, None])[:, 0]
        marg = torch.logaddexp(logp.gather(1, plain[:, None])[:, 0], logp.gather(1, (plain + 260)[:, None])[:, 0])
        n = len(chunk)
        return {"bytes": n, "bits_fused": float(-fused.sum() / math.log(2)),
                "bits_marginal": float(-marg.sum() / math.log(2)), "patches": int(bm.sum())}


def _token_chunk(model: Any, tok: Any, chunk: bytes) -> dict[str, float]:
    import torch

    dev = next(model.parameters()).device
    text = chunk.decode("utf-8", errors="ignore")
    ids = tok(text, return_tensors="pt")["input_ids"][0].tolist()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ids = torch.tensor([[bos] + ids], device=dev)
    with torch.inference_mode():
        logits = model(ids).logits[0, :-1].float()
        nll = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum")
    n = len(text.encode("utf-8"))
    return {"bytes": n, "bits_fused": float(nll / math.log(2)), "bits_marginal": float(nll / math.log(2)), "patches": 0}


def run_bpb(model: Any, tok: Any, cfg: ProfileConfig, limit: int | None) -> dict[str, Any]:
    from .counting import is_bolmo_like

    data = load_wikitext_bytes()
    chunks = chunk_bytes(data)
    if limit:
        chunks = chunks[:limit]
    bolmo = is_bolmo_like(model)
    ms = None
    if bolmo:
        import importlib

        ms = importlib.import_module(type(model.model).__module__).MaskState
    rows = []
    t0 = time.perf_counter()
    for i, c in enumerate(chunks):
        rows.append(_bolmo_chunk(model, tok, c, ms) if bolmo else _token_chunk(model, tok, c))
        if (i + 1) % 25 == 0:
            log.info("bpb chunk %d/%d", i + 1, len(chunks))
    pairs = [(r["bits_fused"], r["bytes"]) for r in rows]
    mean, lo, hi = bootstrap_ci(pairs, ratio_of_sums, seed=cfg.seed)
    pairs_m = [(r["bits_marginal"], r["bytes"]) for r in rows]
    mean_m, lo_m, hi_m = bootstrap_ci(pairs_m, ratio_of_sums, seed=cfg.seed)
    return {"bpb": mean, "bpb_ci95": [lo, hi], "bpb_marginal": mean_m, "bpb_marginal_ci95": [lo_m, hi_m],
            "n_chunks": len(rows), "n_bytes": sum(r["bytes"] for r in rows), "chunk_bytes": BPB_CHUNK,
            "dataset": {"repo": WIKITEXT_REPO, "revision": WIKITEXT_REV, "file": WIKITEXT_FILE,
                        "first_bytes": BPB_BYTES},
            "definition": ("Bolmo: NLL of fused (byte, boundary) tokens / bytes, boundaries from the model's own "
                           "predictor" if bolmo else "token NLL summed / bytes"),
            "seconds": time.perf_counter() - t0, "chunks": rows}


# ---------------------------------------------------------------------------
# CUTE
# ---------------------------------------------------------------------------


def select_cute_items(data: dict[str, list[dict[str, str]]], total: int, seed: int) -> list[tuple[str, int]]:
    sizes = {t: len(v) for t, v in data.items()}
    # more than the original 500: a superset of the original sample, so old and new results compare
    picks = (stratified_sample(sizes, total, seed) if total <= CUTE_ITEMS
             else superset_sample(sizes, total, CUTE_ITEMS, seed))
    return [(t, i) for t in sorted(picks) for i in picks[t]]


def run_cute(model: Any, tok: Any, cfg: ProfileConfig, limit: int | None) -> dict[str, Any]:
    import torch

    from .runner import generate_checked, set_seed

    data = load_cute()
    items = select_cute_items(data, limit or CUTE_ITEMS, CUTE_SEED)
    sync = torch.cuda.synchronize if cfg.device.startswith("cuda") else (lambda: None)
    handle = None
    if cfg.optimize != "none":
        from . import optim

        handle = optim.apply(model, tok, cfg)
    kw = handle.generate_kwargs() if handle else {}
    set_seed(cfg.seed)
    rows, t0 = [], time.perf_counter()
    for k, (task, idx) in enumerate(items):
        item = data[task][idx]
        budget = min(160, len(item["answer"].encode("utf-8")) + 8)  # gold length + slack, same for all variants
        r = generate_checked(model, tok, item["prompt"] + CUTE_SUFFIX, replace(cfg, max_new_bytes=budget), sync,
                             want_text=True, extra_kwargs=handle.generate_kwargs() if handle else None)
        gen = r.text.decode("utf-8", errors="replace")
        rows.append({"task": task, "index": idx, "gold": item["answer"], "generated": gen[:200],
                     "correct": cute_correct(gen, item["answer"])})
        if (k + 1) % 50 == 0:
            log.info("cute %d/%d acc so far %.3f", k + 1, len(items), sum(x["correct"] for x in rows) / len(rows))
    flags = [float(r["correct"]) for r in rows]
    acc, lo, hi = bootstrap_ci(flags, lambda v: sum(v) / len(v), seed=cfg.seed)
    per_task = {t: sum(r["correct"] for r in rows if r["task"] == t) / max(1, sum(r["task"] == t for r in rows))
                for t in sorted({r["task"] for r in rows})}
    if handle:
        handle.close()
    return {"accuracy": acc, "accuracy_ci95": [lo, hi], "n_items": len(rows), "per_task_accuracy": per_task,
            "dataset": {"repo": CUTE_REPO, "revision": CUTE_REV, "sample_seed": CUTE_SEED},
            "prompt_format": "<CUTE prompt>" + CUTE_SUFFIX + " ... answer = text up to the next quote, stripped",
            "seconds": time.perf_counter() - t0, "items": rows}


def evaluate(cfg: ProfileConfig, task: str, limit: int | None, output_dir: Path | None) -> Path:
    from .runner import environment, load_model, set_seed

    model, tok = load_model(cfg)
    set_seed(cfg.seed)
    result = run_bpb(model, tok, cfg, limit) if task == "bpb" else run_cute(model, tok, cfg, limit)
    meta = environment(cfg)
    if hasattr(model, "_byteprof_quant"):
        meta["quantization"] = model._byteprof_quant
    import torch

    meta["gpu_mem_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20 if cfg.device.startswith("cuda") else 0
    out_dir = output_dir or cfg.output_dir / ("eval-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps({"task": task, "limit": limit, "meta": meta, "result": result},
                                                  indent=1, default=str), encoding="utf-8")
    return out_dir / "eval.json"
