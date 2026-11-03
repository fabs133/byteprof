"""Profiling configuration.

Precedence (lowest -> highest): built-in defaults < TOML config file < CLI overrides.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised for invalid or inconsistent configuration."""


@dataclass(frozen=True)
class ProfileConfig:
    model_id: str = "allenai/Bolmo-1B"
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    prompts_file: Path = Path("prompts/default.jsonl")
    max_new_bytes: int = 256
    warmup_runs: int = 2
    measured_runs: int = 5
    seed: int = 1234
    output_dir: Path = Path("results")
    trace: bool = False
    # Dotted module name whose forward() marks one model step (prefill or one
    # decode step). Empty string = the top-level model returned by from_pretrained.
    step_module: str = ""
    # "none" = eager (the measured baseline, never touches byteprof/optim);
    # "compile" = torch.compile(mode="reduce-overhead"); "cuda_graph" = manual CUDA graphs.
    # Optimized variants cannot be hooked, so profile reports wall time only for them.
    optimize: str = "none"
    # Backend for optimize = "compile": "inductor" or "cudagraphs" (graph capture, no codegen).
    compile_backend: str = "inductor"
    # Bolmo only, with optimize = "cuda_graph": also capture the global model (16 layers,
    # static KV cache) so the whole decode step except generate()'s own loop is graphed.
    optimize_global: bool = False
    # Opt-in instrumentation of optimized runs, never part of reported throughput:
    # "timing" = GPU time of graph replays (CUDA events) and host time inside them;
    # "syncs" = count host syncs in the decode loop. Needs optimize != "none".
    instrument: str = "none"
    # "hf" = the model's own generate() loop; "minimal" = byteprof's one-sync-per-token loop
    # (optim/*_loop.py). Needs optimize != "none".
    decode_loop: str = "hf"
    # Weight-only quantization of selected linears, applied right after loading (before any
    # optimization/graph capture): {target = none|local|global|all, scheme = int8_wo|int4_wo,
    # group_size = 128 (int4 only)}. Empty / target = "none" never imports byteprof.quant.
    quantize: dict[str, Any] = field(default_factory=dict)
    # Patch-level drafting (TASK_drafting.md): {kind = none|context|ngram|combined|oracle, k_max, ks, min_len,
    # min_count, max_order, ...}. Empty / kind = "none" leaves the minimal loop unchanged and never imports
    # byteprof.drafting or byteprof.optim.bolmo_draft. Needs decode_loop = "minimal".
    draft: dict[str, Any] = field(default_factory=dict)
    # Component name -> list of regexes matched (re.search) against dotted module names.
    # Order matters: first component whose pattern matches wins.
    components: dict[str, list[str]] = field(default_factory=dict)


_PATH_FIELDS = {"prompts_file", "output_dir"}
_INT_FIELDS = {"max_new_bytes", "warmup_runs", "measured_runs", "seed"}
_VALID_DTYPES = {"bfloat16", "float16", "float32"}
VALID_OPTIMIZE = ("none", "compile", "cuda_graph")
VALID_COMPILE_BACKENDS = ("inductor", "cudagraphs")
VALID_INSTRUMENT = ("none", "timing", "syncs")
VALID_DECODE_LOOPS = ("hf", "minimal")
# Duplicated from byteprof.quant.select on purpose: this module must not import byteprof.quant.
VALID_QUANT_TARGETS = ("none", "local", "global", "all", "blocks", "blocks+head")
VALID_QUANT_SCHEMES = ("int8_wo", "int4_wo")
VALID_QUANT_KERNELS = ("default", "skinny")
SKINNY_ONLY_SCHEMES = ("bf16_wo",)  # bf16 weights through the skinny kernel
VALID_DRAFT_KINDS = ("none", "context", "ngram", "combined", "oracle")
VALID_DRAFT_VERIFY = ("steps", "fused")
VALID_DRAFT_POLICY = ("within", "gated")
DRAFT_KEYS = {"kind", "k_max", "ks", "min_len", "min_count", "max_order", "orders", "corpus_dir", "truth_file",
              "truth_length", "k_ngram", "verify", "policy", "k_cross", "max_patches"}


def _coerce(raw: dict[str, Any], source: str) -> dict[str, Any]:
    known = {f.name for f in fields(ProfileConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{source}: unknown key(s): {', '.join(sorted(unknown))}")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if key in _PATH_FIELDS:
            value = Path(value)
        out[key] = value
    return out


def validate(cfg: ProfileConfig, *, require_components: bool) -> ProfileConfig:
    for name in _INT_FIELDS:
        value = getattr(cfg, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"{name} must be an integer, got {value!r}")
    if cfg.max_new_bytes < 1 or cfg.measured_runs < 1 or cfg.warmup_runs < 0:
        raise ConfigError("max_new_bytes and measured_runs must be >= 1, warmup_runs >= 0")
    if cfg.dtype not in _VALID_DTYPES:
        raise ConfigError(f"dtype must be one of {sorted(_VALID_DTYPES)}, got {cfg.dtype!r}")
    if cfg.optimize not in VALID_OPTIMIZE:
        raise ConfigError(f"optimize must be one of {list(VALID_OPTIMIZE)}, got {cfg.optimize!r}")
    if cfg.instrument not in VALID_INSTRUMENT:
        raise ConfigError(f"instrument must be one of {list(VALID_INSTRUMENT)}, got {cfg.instrument!r}")
    if cfg.decode_loop not in VALID_DECODE_LOOPS:
        raise ConfigError(f"decode_loop must be one of {list(VALID_DECODE_LOOPS)}, got {cfg.decode_loop!r}")
    if cfg.optimize == "none" and (cfg.instrument != "none" or cfg.decode_loop != "hf"):
        raise ConfigError("instrument and decode_loop = 'minimal' need optimize != 'none'")
    if cfg.optimize_global and cfg.optimize != "cuda_graph":
        raise ConfigError("optimize_global requires optimize = 'cuda_graph'")
    q = cfg.quantize
    if not isinstance(q, dict) or set(q) - {"target", "scheme", "group_size", "local_scheme", "global_scheme", "kernel"}:
        raise ConfigError("quantize must be a table with keys target, scheme, group_size, local_scheme, global_scheme, kernel")
    if q.get("kernel", "default") not in VALID_QUANT_KERNELS:
        raise ConfigError(f"quantize.kernel must be one of {list(VALID_QUANT_KERNELS)}, got {q.get('kernel')!r}")
    schemes = VALID_QUANT_SCHEMES + (SKINNY_ONLY_SCHEMES if q.get("kernel") == "skinny" else ())
    if q.get("target", "none") not in VALID_QUANT_TARGETS:
        raise ConfigError(f"quantize.target must be one of {list(VALID_QUANT_TARGETS)}, got {q.get('target')!r}")
    if q.get("target", "none") != "none":
        if q.get("scheme") not in schemes:
            raise ConfigError(f"quantize.scheme must be one of {list(schemes)}, got {q.get('scheme')!r}")
        for key in ("local_scheme", "global_scheme"):
            if key in q and q[key] not in schemes:
                raise ConfigError(f"quantize.{key} must be one of {list(schemes)}, got {q[key]!r}")
        gs = q.get("group_size", 128)
        if q.get("kernel") == "skinny" and "int4_wo" in (q["scheme"], q.get("local_scheme"), q.get("global_scheme")) and gs != 128:
            raise ConfigError("quantize.group_size must be 128 for int4_wo with kernel = skinny")
        if "int4_wo" in (q["scheme"], q.get("local_scheme"), q.get("global_scheme")) and (not isinstance(gs, int) or isinstance(gs, bool) or gs not in (32, 64, 128, 256)):
            raise ConfigError(f"quantize.group_size must be one of 32, 64, 128, 256 for int4_wo, got {gs!r}")
    dr = cfg.draft
    if not isinstance(dr, dict) or set(dr) - DRAFT_KEYS:
        raise ConfigError(f"draft must be a table with keys {sorted(DRAFT_KEYS)}")
    if dr.get("kind", "none") not in VALID_DRAFT_KINDS:
        raise ConfigError(f"draft.kind must be one of {list(VALID_DRAFT_KINDS)}, got {dr.get('kind')!r}")
    if dr.get("verify", "steps") not in VALID_DRAFT_VERIFY:
        raise ConfigError(f"draft.verify must be one of {list(VALID_DRAFT_VERIFY)}, got {dr.get('verify')!r}")
    if dr.get("policy", "within") not in VALID_DRAFT_POLICY:
        raise ConfigError(f"draft.policy must be one of {list(VALID_DRAFT_POLICY)}, got {dr.get('policy')!r}")
    if dr.get("policy", "within") == "gated":
        if dr.get("kind", "none") not in ("context", "ngram", "combined"):
            raise ConfigError("draft.policy = 'gated' needs a lookup drafter (draft.kind context, ngram or combined)")
        kc, mp = dr.get("k_cross", 32), dr.get("max_patches", 8)
        if (not isinstance(kc, int) or isinstance(kc, bool) or not 1 <= kc <= 32
                or not isinstance(mp, int) or isinstance(mp, bool) or not 1 <= mp <= 8):
            raise ConfigError("draft.k_cross must be an integer in 1..32 and draft.max_patches an integer in 1..8")
    if dr.get("kind", "none") != "none":
        if cfg.decode_loop != "minimal":
            raise ConfigError("draft.kind != 'none' needs decode_loop = 'minimal'")
        ks = dr.get("ks", [0, 1, 2, 4, 6, 8])
        km = dr.get("k_max", max(ks) if ks else 0)
        if (not isinstance(ks, list) or not all(isinstance(k, int) and not isinstance(k, bool) and 0 <= k <= 32 for k in ks)
                or not isinstance(km, int) or isinstance(km, bool) or not 1 <= km <= 32):
            raise ConfigError("draft.ks must be a list of integers in 0..32 and draft.k_max an integer in 1..32")
        if dr["kind"] == "oracle" and "truth_file" not in dr:
            raise ConfigError("draft.kind = 'oracle' needs draft.truth_file (the recorded ground truth)")
    if cfg.compile_backend not in VALID_COMPILE_BACKENDS:
        raise ConfigError(
            f"compile_backend must be one of {list(VALID_COMPILE_BACKENDS)}, got {cfg.compile_backend!r}"
        )
    if not isinstance(cfg.components, dict) or not all(
        isinstance(v, list) and all(isinstance(p, str) for p in v) for v in cfg.components.values()
    ):
        raise ConfigError("components must be a table of name = [regex, ...]")
    if require_components and not cfg.components:
        raise ConfigError(
            "no [components] configured. Run `python -m byteprof discover` first "
            "and fill the [components] table in your config file."
        )
    return cfg


def load_config(
    path: Path | None,
    overrides: dict[str, Any] | None = None,
    *,
    require_components: bool = False,
) -> ProfileConfig:
    cfg = ProfileConfig()
    if path is not None:
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
        cfg = replace(cfg, **_coerce(raw, str(path)))
    if overrides:
        cfg = replace(cfg, **_coerce(overrides, "command line"))
    return validate(cfg, require_components=require_components)
