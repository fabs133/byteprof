"""Writes configs/quant/*.toml: Stage 4 variants of TASK_local_quant.md (same prompts/seed/loop as bf16)."""

from pathlib import Path

BOLMO = """# Bolmo 1B, graph-global + minimal loop, {desc}
model_id = "allenai/Bolmo-1B"
revision = "452690a8cfab1dd1ec27b74f2667632f17a21b46"
dtype = "bfloat16"
device = "cuda"
prompts_file = "prompts/default.jsonl"
max_new_bytes = 256
warmup_runs = 10
measured_runs = 5
seed = 1234
output_dir = "results/quant/bolmo-{name}"
optimize = "cuda_graph"
optimize_global = true
decode_loop = "minimal"
compile_backend = "inductor"
{quant}"""

OLMO = """# OLMo 2 1B, compiled + minimal loop, {desc}
model_id = "allenai/OLMo-2-0425-1B"
revision = "a1847dff35000b4271fa70afc5db10fd29fedbdf"
dtype = "bfloat16"
device = "cuda"
prompts_file = "prompts/default.jsonl"
max_new_bytes = 72
warmup_runs = 10
measured_runs = 5
seed = 1234
output_dir = "results/quant/olmo-{name}"
optimize = "compile"
decode_loop = "minimal"
compile_backend = "inductor"
{quant}"""

VARIANTS = {
    "bf16": ("bf16 (no quantization)", None),
    "local-int8": ("local stack int8 weight-only", ("local", "int8_wo")),
    "local-int4": ("local stack int4 weight-only, group 128", ("local", "int4_wo")),
    "global-int4": ("global layers int4 weight-only, group 128", ("global", "int4_wo")),
    "all-int4": ("local + global int4 weight-only, group 128", ("all", "int4_wo")),
}


def quant_block(spec):
    if spec is None:
        return ""
    target, scheme = spec
    gs = "group_size = 128\n" if scheme == "int4_wo" else ""
    return f'\n[quantize]\ntarget = "{target}"\nscheme = "{scheme}"\n{gs}'


out = Path(__file__).resolve().parent.parent / "configs" / "quant"
out.mkdir(parents=True, exist_ok=True)
for name, (desc, spec) in VARIANTS.items():
    (out / f"bolmo-{name}.toml").write_text(BOLMO.format(name=name, desc=desc, quant=quant_block(spec)), encoding="utf-8")
for name in ("bf16", "all-int4"):
    desc, spec = VARIANTS[name]
    (out / f"olmo-{name}.toml").write_text(OLMO.format(name=name, desc=desc, quant=quant_block(spec)), encoding="utf-8")
print(sorted(p.name for p in out.glob("*.toml")))
