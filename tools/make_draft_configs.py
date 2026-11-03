"""Writes configs/draft/bolmo-{bf16,int4,fp32}-{none,context,ngram,combined,oracle}.toml (TASK_drafting.md).

none = the unchanged minimal loop (same settings as configs/fair, so the comparison is the same loop with and
without the [draft] table). Drafter parameters are the ones chosen on the extended prompt set in Stage 0.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEAD = """# Bolmo 1B, graph-global + minimal loop{what}
model_id = "allenai/Bolmo-1B"
revision = "452690a8cfab1dd1ec27b74f2667632f17a21b46"
dtype = "{dtype}"
device = "cuda"
prompts_file = "prompts/default.jsonl"
max_new_bytes = 256
warmup_runs = 10
measured_runs = 5
seed = 1234
output_dir = "results/drafting/bolmo-{prec}-{kind}"
optimize = "cuda_graph"
optimize_global = true
decode_loop = "minimal"
compile_backend = "inductor"
"""
QUANT = """
[quantize]
target = "all"
scheme = "int4_wo"
group_size = 128
"""
DRAFT = """
[draft]
kind = "{kind}"
ks = [0, 1, 2, 4, 6, 8]
k_max = 8
min_len = 3
min_count = 1
max_order = 6
"""
ORACLE = """truth_file = "results/drafting/truth/{prec}.json"
truth_length = 1024
"""

out = ROOT / "configs" / "draft"
out.mkdir(parents=True, exist_ok=True)
for prec, dtype in (("bf16", "bfloat16"), ("int4", "bfloat16"), ("fp32", "float32")):
    for kind in ("none", "context", "ngram", "combined", "oracle"):
        s = HEAD.format(what=f", {prec}, drafting kind = {kind}", dtype=dtype, prec=prec, kind=kind)
        if prec == "int4":
            s += QUANT
        if kind != "none":
            s += DRAFT.format(kind=kind) + (ORACLE.format(prec=prec) if kind == "oracle" else "")
        (out / f"bolmo-{prec}-{kind}.toml").write_text(s, encoding="utf-8")
# Stage 3 speed cells: (precision, kind, k_max)
CELLS = [("bf16", "combined", 6), ("bf16", "oracle", 6), ("bf16", "combined", 4), ("bf16", "combined", 8),
         ("bf16", "context", 4), ("int4", "combined", 4), ("int4", "oracle", 4), ("int4", "combined", 2),
         ("int4", "combined", 6), ("int4", "context", 4)]
for prec, kind, km in CELLS:
    dtype = "float32" if prec == "fp32" else "bfloat16"
    s = HEAD.format(what=f", {prec}, drafting kind = {kind}, k_max = {km}", dtype=dtype, prec=prec, kind=f"{kind}-k{km}")
    if prec == "int4":
        s += QUANT
    s += DRAFT.format(kind=kind).replace("k_max = 8", f"k_max = {km}")
    if kind == "oracle":  # self-consistent truth from tools/draft_oracle_truth.py (fixed point of this very loop)
        s += f'truth_file = "results/drafting/truth/oracle-{prec}-k{km}.json"\ntruth_length = 256\n'
    (out / f"bolmo-{prec}-{kind}-k{km}.toml").write_text(s, encoding="utf-8")
print(sorted(p.name for p in out.glob("*.toml")))
