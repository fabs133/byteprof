"""Writes configs/p4/*.toml (TASK_build_p4.md): boundary-gated lookup drafting (P4) and its comparison cells.

Cells: P4 (draft.policy = "gated": within-patch rounds at k_max, crossing rounds at k_cross, at most max_patches patches)
and, for the comparison, P0 (policy "within", fused verification, the best K of Task A) in bf16 and all-int4. Every cell
uses the Task A fused verification and the same ks.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEAD = """# Bolmo 1B, graph-global + minimal loop, {what}
model_id = "allenai/Bolmo-1B"
revision = "452690a8cfab1dd1ec27b74f2667632f17a21b46"
dtype = "{dtype}"
device = "cuda"
prompts_file = "prompts/default.jsonl"
max_new_bytes = 256
warmup_runs = 10
measured_runs = 5
seed = 1234
output_dir = "results/p4/{name}"
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
kind = "combined"
ks = [0, 1, 2, 4, 6, 8, 12, 16, 24, 32]
k_max = {k_max}
min_len = 3
min_count = 1
max_order = 6
verify = "fused"
"""
GATED = """policy = "gated"
k_cross = {k_cross}
max_patches = {max_patches}
"""

out = ROOT / "configs" / "p4"
out.mkdir(parents=True, exist_ok=True)


def write(name: str, prec: str, what: str, draft: dict | None) -> None:
    dtype = "float32" if prec == "fp32" else "bfloat16"
    s = HEAD.format(what=what, dtype=dtype, name=name)
    if prec == "int4":
        s += QUANT
    if draft is not None:
        s += DRAFT.format(k_max=draft["k_max"])
        if draft.get("policy") == "gated":
            s += GATED.format(k_cross=draft["k_cross"], max_patches=draft.get("max_patches", 8))
    (out / f"{name}.toml").write_text(s, encoding="utf-8")


# P4 cells: (k_cross, k_max) chosen on the extended set by tools/p4_costmodel.py with the measured Stage 0 cost:
# bf16 32 / 6, int4 16 / 4 (the task text names 32 and 12; 12 / 4 is written as an extra cell)
for prec, k_within, k_cross in (("bf16", 6, 32), ("int4", 4, 16), ("int4", 4, 12), ("fp32", 6, 32)):
    write(f"{prec}-p4" + (f"-kc{k_cross}" if (prec, k_cross) == ("int4", 12) else ""), prec, f"{prec}, boundary-gated lookup drafting (P4), k_max {k_within}, k_cross {k_cross}",
          {"policy": "gated", "k_max": k_within, "k_cross": k_cross})
# P0 cells (fused verification): best K of Task A
for prec, k in (("bf16", 6), ("bf16", 8), ("int4", 4), ("int4", 6)):
    write(f"{prec}-p0-k{k}", prec, f"{prec}, within-patch lookup drafting (P0), k_max {k}, fused", {"policy": "within", "k_max": k})
# the minimal loop (no drafting)
for prec in ("bf16", "int4"):
    write(f"{prec}-none", prec, f"{prec}, minimal loop", None)
# microbenchmark cells: all crossing graphs, bf16 and int4
for prec in ("bf16", "int4"):
    write(f"{prec}-bench", prec, f"{prec}, crossing-round microbenchmark (P4 graphs)", {"policy": "gated", "k_max": 8, "k_cross": 32})
# extended prompt set, one pass each (acceptance: bytes per round / per global pass, share of crossing rounds)
for prec, k_within, k_cross in (("bf16", 6, 32), ("int4", 4, 16)):
    write(f"{prec}-p4-ext", prec, f"{prec}, P4 on the extended prompt set (acceptance)",
          {"policy": "gated", "k_max": k_within, "k_cross": k_cross})
    f = out / f"{prec}-p4-ext.toml"
    t = f.read_text(encoding="utf-8").replace('prompts_file = "prompts/default.jsonl"', 'prompts_file = "prompts/drafting_ext.jsonl"')
    t = t.replace("measured_runs = 5", "measured_runs = 1")
    f.write_text(t, encoding="utf-8")
print(sorted(p.name for p in out.glob("*.toml")))
