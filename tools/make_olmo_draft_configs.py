"""Writes configs/draft/olmo-bf16-*.toml (OLMo 2 token drafting; parameters from results/drafting/stage3/olmo_sim.json)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
chosen = json.loads((ROOT / "results/drafting/stage3/olmo_sim.json").read_text(encoding="utf-8"))["chosen"]
HEAD = """# OLMo 2 1B, static KV cache + manual CUDA graph + minimal loop, {dtype}{what}
model_id = "allenai/OLMo-2-0425-1B"
revision = "a1847dff35000b4271fa70afc5db10fd29fedbdf"
dtype = "{dtype}"
device = "cuda"
prompts_file = "prompts/default.jsonl"
max_new_bytes = 72
warmup_runs = 10
measured_runs = 5
seed = 1234
output_dir = "results/drafting/olmo-{prec}-{cell}"
optimize = "cuda_graph"
decode_loop = "minimal"
"""
out = ROOT / "configs" / "draft"
cells = [("none", None, 0)] + [(f"{kind}-k{k}", kind, k) for kind in ("context", "combined", "ngram") for k in (1, 2, 4, 6, 8)]
for prec, dtype in (("bf16", "bfloat16"), ("fp32", "float32")):
    for cell, kind, k in cells:
        s = HEAD.format(dtype=dtype, what=f", {cell}", prec=prec, cell=cell)
        if kind:
            c = chosen["combined"]
            ml = chosen["context_min_len"] if kind == "context" else c["min_len"]
            mc = chosen["ngram_min_count"] if kind == "ngram" else c["min_count"]
            mo = chosen["ngram_max_order"] if kind == "ngram" else c["max_order"]
            s += f'\n[draft]\nkind = "{kind}"\nks = [0, 1, 2, 4, 6, 8]\nk_max = {k}\nmin_len = {ml}\nmin_count = {mc}\nmax_order = {mo}\n'
        (out / f"olmo-{prec}-{cell}.toml").write_text(s, encoding="utf-8")
print(sorted(p.name for p in out.glob("olmo-*.toml")))
