"""Model loading, generation and the profiling workflow (the IO boundary)."""

from __future__ import annotations

import json
import logging
import platform
import random
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .components import ComponentMatch, match_components, unmatched_components
from .config import ProfileConfig
from .counting import AlignmentError, cap_for, new_bytes, unit_of
from .report import render_markdown, render_optimized_markdown, to_json
from .timing import STEP, CpuClock, CudaEventClock, HookTimer, Span, summarize

log = logging.getLogger("byteprof")


class ModelLoadError(Exception):
    """Raised when the model or tokenizer cannot be loaded."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_prompts(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"prompts file not found: {path}")
    prompts = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj.get("text"), str) or not obj["text"]:
            raise ValueError(f"{path}:{lineno}: each line needs a non-empty 'text' field")
        prompts.append({"id": str(obj.get("id", lineno)), "text": obj["text"]})
    if not prompts:
        raise ValueError(f"{path}: no prompts")
    return prompts


# ---------------------------------------------------------------------------
# Model adapter
# ---------------------------------------------------------------------------


def load_model(cfg: ProfileConfig) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ModelLoadError(f"missing dependency: {exc}. Install requirements.txt.") from exc
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise ModelLoadError("device is 'cuda' but torch.cuda.is_available() is False")
    dtype = getattr(torch, cfg.dtype)
    try:
        tok = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, revision=cfg.revision, torch_dtype=dtype, trust_remote_code=True
        )
    except Exception as exc:  # remote code can raise anything
        raise ModelLoadError(f"could not load {cfg.model_id}: {type(exc).__name__}: {exc}") from exc
    model.to(cfg.device).eval()
    if cfg.quantize.get("target", "none") != "none":  # opt-in: quantization code is imported only here
        from .quant.integrate import apply_quantization

        q = cfg.quantize
        model._byteprof_quant = apply_quantization(model, q["target"], q["scheme"], q.get("group_size", 128),
                                                       q.get("local_scheme"), q.get("global_scheme"), q.get("kernel", "default"))
        log.info("quantized: %s", model._byteprof_quant)
    return model, tok


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class GenResult:
    new_bytes: int
    ms: float
    prompt_bytes: int
    input_len: int
    output_len: int
    unit: str  # unit of max_new_bytes for this model: "bytes" or "tokens"
    cap: int  # largest plausible number of new bytes for this run
    text: bytes = b""  # generated UTF-8 bytes (only filled when want_text=True)


class _MaxAtoms:
    """StoppingCriteria for Bolmo: stop after `n` new bytes.

    Bolmo's generate() hands criteria the list of generated atoms, which starts with
    the second-to-last prompt byte and the re-decoded last prompt byte (n + 2 atoms
    = n new bytes). Its own max_new_tokens check is relative to the list start, so it
    allows about input_len + max_new_tokens new bytes; this criterion enforces n.
    """

    def __new__(cls, n: int):
        import torch
        from transformers import StoppingCriteria

        class Impl(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full((input_ids.shape[0],), input_ids.shape[1] >= n + 2, dtype=torch.bool,
                                  device=input_ids.device)

        crit = Impl()
        crit.n = n  # read by byteprof.optim's minimal decode loop
        return crit


_MAX_TOKEN_BYTES: dict[int, int] = {}


def _max_token_bytes(tok: Any) -> int:
    key = id(tok)
    if key not in _MAX_TOKEN_BYTES:
        vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
        # byte-level BPE vocab strings use one char per byte; other tokenizers are approximate
        _MAX_TOKEN_BYTES[key] = max((len(t.encode("utf-8")) for t in vocab), default=32)
    return _MAX_TOKEN_BYTES[key]


def generate_checked(model: Any, tok: Any, prompt: str, cfg: ProfileConfig, sync: Callable[[], None],
                     *, want_text: bool = False, extra_kwargs: dict[str, Any] | None = None) -> GenResult:
    """Greedy generation of at most cfg.max_new_bytes (bytes for Bolmo, tokens otherwise)."""
    import torch

    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(cfg.device)
    unit = unit_of(model)
    kwargs: dict[str, Any] = {"max_new_tokens": cfg.max_new_bytes, "do_sample": False}
    if unit == "bytes":
        from transformers import StoppingCriteriaList

        kwargs["stopping_criteria"] = StoppingCriteriaList([_MaxAtoms(cfg.max_new_bytes)])
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(input_ids, **kwargs)
    sync()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    decode = lambda ids: tok.decode(list(ids), skip_special_tokens=True)  # noqa: E731
    n = new_bytes(input_ids[0].tolist(), out[0].tolist(), decode)
    cap = cap_for(unit, cfg.max_new_bytes, max_token_bytes=_max_token_bytes(tok) if unit == "tokens" else 1)
    text = b""
    if want_text:
        prompt_len = len(decode(input_ids[0].tolist()).encode("utf-8"))
        text = decode(out[0].tolist()).encode("utf-8")[prompt_len:]
    return GenResult(n, wall_ms, len(prompt.encode("utf-8")), input_ids.shape[-1], out.shape[-1], unit, cap, text)


def generate_once(model: Any, tok: Any, prompt: str, cfg: ProfileConfig, sync: Callable[[], None]) -> tuple[int, float]:
    """Returns (generated UTF-8 bytes, wall ms)."""
    r = generate_checked(model, tok, prompt, cfg, sync)
    return r.new_bytes, r.ms


def resolve_module(model: Any, dotted: str) -> Any:
    if not dotted:
        return model
    try:
        return model.get_submodule(dotted)
    except AttributeError as exc:
        raise ValueError(f"step_module '{dotted}' not found in model") from exc


def describe_modules(model: Any, max_depth: int) -> list[str]:
    rows = []
    for name, mod in model.named_modules():
        depth = 0 if not name else name.count(".") + 1
        if depth > max_depth:
            continue
        n_params = sum(p.numel() for p in mod.parameters())
        rows.append(f"{'  ' * depth}{name or '<root>'}  [{type(mod).__name__}]  params={n_params:,}")
    return rows


def suggest_patterns(module_names: list[str]) -> dict[str, list[str]]:
    """Keyword heuristics only - a starting point, always verify against the tree."""
    keywords = {
        "boundary_predictor": ["boundar"],
        "local_encoder": ["local_encoder", "encoder"],
        "local_decoder": ["local_decoder", "decoder"],
        "lm_head": ["lm_head"],
        "global_model": ["global", "backbone"],
    }
    out: dict[str, list[str]] = {}
    for comp, kws in keywords.items():
        hits = [n for n in module_names if any(k in n.split(".")[-1].lower() for k in kws)]
        if hits:
            shallowest = min(hits, key=lambda n: n.count("."))
            out[comp] = ["^" + re.escape(shallowest) + "$"]
    return out


def environment(cfg: ProfileConfig) -> dict[str, Any]:
    import torch
    import transformers

    env: dict[str, Any] = {
        "model_id": cfg.model_id,
        "revision": cfg.revision,
        "dtype": cfg.dtype,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {k: v for k, v in asdict(cfg).items()},
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return env


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------


def discover(cfg: ProfileConfig, max_depth: int = 4) -> Path:
    model, _ = load_model(cfg)
    names = [n for n, _ in model.named_modules()]
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_path = out_dir / "module_tree.txt"
    lines = describe_modules(model, max_depth)
    lines += ["", "# Suggested [components] (heuristic - VERIFY):"]
    for comp, pats in suggest_patterns(names).items():
        lines.append(f'{comp} = {json.dumps(pats)}')
    tree_path.write_text("\n".join(lines), encoding="utf-8")
    return tree_path


def _check_matches(model: Any, matches: list[ComponentMatch], cfg: ProfileConfig) -> list[str]:
    import torch.nn as nn

    problems = []
    for comp in unmatched_components(matches, cfg.components):
        problems.append(f"component '{comp}' matched no module")
    for m in matches:
        mod = model.get_submodule(m.module_name)
        if isinstance(mod, (nn.ModuleList, nn.ModuleDict)):
            problems.append(
                f"'{m.module_name}' ({m.component}) is a {type(mod).__name__}; it has no forward(), "
                "so its hooks never fire. Match its children or its parent instead."
            )
    return problems


def profile(cfg: ProfileConfig) -> Path:
    if cfg.optimize != "none":
        return profile_optimized(cfg)
    import torch

    prompts = load_prompts(cfg.prompts_file)
    model, tok = load_model(cfg)
    use_cuda = cfg.device.startswith("cuda")
    clock = CudaEventClock() if use_cuda else CpuClock()
    sync = torch.cuda.synchronize if use_cuda else (lambda: None)

    names = [n for n, _ in model.named_modules()]
    matches = match_components(names, cfg.components)
    problems = _check_matches(model, matches, cfg)
    if problems:
        raise ValueError("component configuration problems:\n  - " + "\n  - ".join(problems))
    for m in matches:
        log.info("component %-20s <- %s", m.component, m.module_name)
    step_module = resolve_module(model, cfg.step_module)

    set_seed(cfg.seed)
    log.info("warmup: %d run(s)", cfg.warmup_runs)
    for i in range(cfg.warmup_runs):
        generate_once(model, tok, prompts[i % len(prompts)]["text"], cfg, sync)

    workload = [p for _ in range(cfg.measured_runs) for p in prompts]

    # Baseline without hooks: true throughput and hook overhead.
    set_seed(cfg.seed)
    baseline_ms = 0.0
    for p in workload:
        baseline_ms += generate_once(model, tok, p["text"], cfg, sync)[1]

    # Hooked runs.
    set_seed(cfg.seed)
    spans: list[Span] = []
    cap_violations: list[str] = []
    total_bytes, wall_ms = 0, 0.0
    modules = [(m.component, model.get_submodule(m.module_name)) for m in matches]
    with HookTimer(step_module, modules, clock) as timer:
        for run, p in enumerate(workload):
            timer.start_run(run)
            r = generate_checked(model, tok, p["text"], cfg, sync)
            n_bytes, ms = r.new_bytes, r.ms
            total_bytes += n_bytes
            wall_ms += ms
            spans.extend(timer.collect())
            log.info("run %d/%d prompt=%s bytes=%d %.1f ms", run + 1, len(workload), p["id"], n_bytes, ms)
            if n_bytes > r.cap:
                msg = (f"prompt {p['id']}: {n_bytes} new bytes exceed the cap of {r.cap} implied by "
                       f"max_new_bytes={cfg.max_new_bytes} ({r.unit})")
                log.error(msg)
                cap_violations.append(msg)

    bd = summarize(spans, runs=len(workload), generated_bytes=total_bytes, wall_ms=wall_ms)
    bd.warnings.extend(f"byte count: {m}" for m in cap_violations)
    meta = environment(cfg)
    meta["matches"] = [asdict(m) for m in matches]

    out_dir = cfg.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "breakdown.md").write_text(render_markdown(bd, meta, baseline_ms), encoding="utf-8")
    (out_dir / "breakdown.json").write_text(to_json(bd, meta, baseline_ms), encoding="utf-8")

    if cfg.trace:
        _trace(model, tok, prompts[0]["text"], cfg, sync, out_dir)
    return out_dir


def profile_optimized(cfg: ProfileConfig) -> Path:
    """Wall-time profile of an optimized variant (no hooks: graphs bypass them).

    Eager reference outputs are generated first, on the same model object, before the
    optimization is installed; every measured output is compared against them.
    """
    import torch

    from .counting import first_divergence

    prompts = load_prompts(cfg.prompts_file)
    model, tok = load_model(cfg)
    use_cuda = cfg.device.startswith("cuda")
    sync = torch.cuda.synchronize if use_cuda else (lambda: None)

    set_seed(cfg.seed)
    reference = {p["id"]: generate_checked(model, tok, p["text"], cfg, sync, want_text=True).text for p in prompts}
    sync()
    mem_before = torch.cuda.memory_allocated() if use_cuda else 0
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    from . import optim  # imported only for optimized variants

    handle = optim.apply(model, tok, cfg)
    warm_passes = max(2, -(-cfg.warmup_runs // len(prompts)))
    warm_times: list[float] = []
    for _ in range(warm_passes):
        t0 = time.perf_counter()
        for p in prompts:
            generate_checked(model, tok, p["text"], cfg, sync, extra_kwargs=handle.generate_kwargs())
        warm_times.append(time.perf_counter() - t0)
    mem_after = torch.cuda.memory_allocated() if use_cuda else 0
    peak = torch.cuda.max_memory_allocated() if use_cuda else 0

    set_seed(cfg.seed)
    workload = [p for _ in range(cfg.measured_runs) for p in prompts]
    rows: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    cap_violations: list[str] = []
    for run, p in enumerate(workload):
        handle.begin()
        r = generate_checked(model, tok, p["text"], cfg, sync, want_text=True, extra_kwargs=handle.generate_kwargs())
        instr = handle.end()
        outputs.setdefault(p["id"], r.text.decode("utf-8", errors="replace"))
        div = first_divergence(reference[p["id"]], r.text)
        rows.append({"run": run, "prompt_id": p["id"], "new_bytes": r.new_bytes, "ms": r.ms,
                     "first_divergence": div, "ref_len": len(reference[p["id"]]), "out_len": len(r.text),
                     **({"instr": instr} if instr else {})})
        log.info("run %d/%d prompt=%s bytes=%d %.1f ms divergence=%s", run + 1, len(workload), p["id"],
                 r.new_bytes, r.ms, div)
        if r.new_bytes > r.cap:
            cap_violations.append(f"prompt {p['id']}: {r.new_bytes} new bytes exceed the cap of {r.cap}")

    meta = environment(cfg)
    meta["optimization"] = {
        "variant": cfg.optimize, "details": handle.describe(),
        "warmup_pass_seconds": warm_times,
        "gpu_mem_allocated_delta_mib": (mem_after - mem_before) / 2**20,
        "gpu_mem_peak_mib": peak / 2**20,
    }
    if hasattr(model, "_byteprof_quant"):
        meta["quantization"] = model._byteprof_quant
    out_dir = cfg.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "outputs.json").write_text(json.dumps(outputs, indent=1), encoding="utf-8")
    (out_dir / "reference_outputs.json").write_text(
        json.dumps({k: v.decode("utf-8", errors="replace") for k, v in reference.items()}, indent=1), encoding="utf-8")
    (out_dir / "breakdown.md").write_text(render_optimized_markdown(rows, meta, cap_violations), encoding="utf-8")
    (out_dir / "breakdown.json").write_text(
        json.dumps({"rows": rows, "meta": meta, "warnings": cap_violations}, indent=2, default=str), encoding="utf-8")
    if cfg.trace:
        _trace(model, tok, prompts[0]["text"], cfg, sync, out_dir, handle.generate_kwargs())
    handle.close()
    return out_dir


def _trace(model: Any, tok: Any, prompt: str, cfg: ProfileConfig, sync: Callable[[], None], out_dir: Path,
           extra_kwargs: dict[str, Any] | None = None) -> None:
    """One run under torch.profiler: kernel counts and launch overhead."""
    from torch.profiler import ProfilerActivity, profile as tprofile

    activities = [ProfilerActivity.CPU]
    if cfg.device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    with tprofile(activities=activities, record_shapes=False) as prof:
        generate_checked(model, tok, prompt, cfg, sync, extra_kwargs=extra_kwargs)
    prof.export_chrome_trace(str(out_dir / "trace.json"))
    sort_key = "cuda_time_total" if cfg.device.startswith("cuda") else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=40)
    (out_dir / "kernels.txt").write_text(table, encoding="utf-8")


def inspect(cfg: ProfileConfig) -> tuple[Path, list[dict[str, Any]]]:
    """One greedy generation per prompt, no warmup; checks byte counting. Writes inspect.json."""
    import torch

    prompts = load_prompts(cfg.prompts_file)
    model, tok = load_model(cfg)
    use_cuda = cfg.device.startswith("cuda")
    clock = CudaEventClock() if use_cuda else CpuClock()
    sync = torch.cuda.synchronize if use_cuda else (lambda: None)
    step_module = resolve_module(model, cfg.step_module)

    set_seed(cfg.seed)
    rows: list[dict[str, Any]] = []
    with HookTimer(step_module, [], clock) as timer:
        for i, p in enumerate(prompts):
            timer.start_run(i)
            try:
                r = generate_checked(model, tok, p["text"], cfg, sync)
            except AlignmentError as exc:
                log.error("prompt %s: %s", p["id"], exc)
                timer.collect()
                rows.append({"prompt_id": p["id"], "prompt_bytes": len(p["text"].encode("utf-8")),
                             "input_len": 0, "output_len": 0, "new_bytes": 0, "step_calls": 0,
                             "max_new": cfg.max_new_bytes, "ok": False})
                continue
            steps = sum(1 for s in timer.collect() if s.component == STEP)
            rows.append({"prompt_id": p["id"], "prompt_bytes": r.prompt_bytes, "input_len": r.input_len,
                         "output_len": r.output_len, "new_bytes": r.new_bytes, "step_calls": steps,
                         "max_new": cfg.max_new_bytes, "ok": 0 < r.new_bytes <= r.cap})
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inspect.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return out_dir / "inspect.json", rows
