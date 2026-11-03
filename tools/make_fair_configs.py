"""Writes configs/fair/*.toml for TASK_fair_ratio_floor.md (same prompts/seed/loop as the quant task)."""

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
output_dir = "results/fair/bolmo-{name}"
optimize = "cuda_graph"
optimize_global = true
decode_loop = "minimal"
compile_backend = "inductor"
{quant}"""

OLMO = """# OLMo 2 1B, static KV cache + manual CUDA graph (no torch.compile) + minimal loop, {desc}
model_id = "allenai/OLMo-2-0425-1B"
revision = "a1847dff35000b4271fa70afc5db10fd29fedbdf"
dtype = "bfloat16"
device = "cuda"
prompts_file = "prompts/default.jsonl"
max_new_bytes = 72
warmup_runs = 10
measured_runs = 5
seed = 1234
output_dir = "results/fair/olmo-{name}"
optimize = "cuda_graph"
decode_loop = "minimal"
{quant}"""


def q(target=None, scheme=None, **extra):
    if target is None:
        return ""
    lines = ["", "[quantize]", f'target = "{target}"', f'scheme = "{scheme}"']
    if "int4_wo" in (scheme, *extra.values()):
        lines.append("group_size = 128")
    lines += [f'{k} = "{v}"' for k, v in extra.items()]
    return "\n".join(lines) + "\n"


BOLMO_V = {
    "bf16": ("bf16", q()),
    "local-int4": ("local int4", q("local", "int4_wo")),
    "local-int8": ("local int8", q("local", "int8_wo")),
    "global-int4": ("global int4", q("global", "int4_wo")),
    "all-int4": ("local + global int4", q("all", "int4_wo")),
    "global-int8": ("global int8", q("global", "int8_wo")),
    "all-int8": ("local + global int8", q("all", "int8_wo")),
    "l4g8": ("local int4 + global int8", q("all", "int4_wo", global_scheme="int8_wo")),
}
OLMO_V = {
    "bf16": ("bf16", q()),
    "int4-blocks": ("int4 transformer blocks", q("blocks", "int4_wo")),
    "int4-blocks-head": ("int4 blocks + lm_head", q("blocks+head", "int4_wo")),
    "int8-blocks": ("int8 transformer blocks", q("blocks", "int8_wo")),
}
out = Path(__file__).resolve().parent.parent / "configs" / "fair"
out.mkdir(parents=True, exist_ok=True)
for n, (d, quant) in BOLMO_V.items():
    (out / f"bolmo-{n}.toml").write_text(BOLMO.format(name=n, desc=d, quant=quant), encoding="utf-8")
for n, (d, quant) in OLMO_V.items():
    (out / f"olmo-{n}.toml").write_text(OLMO.format(name=n, desc=d, quant=quant), encoding="utf-8")
print(sorted(p.name for p in out.glob("*.toml")))
