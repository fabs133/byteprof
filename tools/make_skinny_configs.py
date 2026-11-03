"""Writes configs/skinny/*.toml (TASK_skinny_kernels.md, Stage 2) from the P4 / drafting templates: Bolmo variants with the
skinny kernels (minimal loop `none` and boundary-gated drafting `p4`) and OLMo 2 with int8 skinny blocks (minimal `none`,
combined lookup `k8`). Usage: python tools/make_skinny_configs.py"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs/skinny"
OUT.mkdir(parents=True, exist_ok=True)

# name -> ([quantize] lines, P4 k_max, P4 k_cross)
BOLMO = {
    "bf16sk": ('target = "all"\nscheme = "bf16_wo"\nkernel = "skinny"', 6, 32),
    "g8": ('target = "all"\nscheme = "bf16_wo"\nglobal_scheme = "int8_wo"\nkernel = "skinny"', 6, 32),
    "g8l4": ('target = "all"\nscheme = "int4_wo"\nglobal_scheme = "int8_wo"\ngroup_size = 128\nkernel = "skinny"', 6, 32),
    "all4": ('target = "all"\nscheme = "int4_wo"\ngroup_size = 128\nkernel = "skinny"', 4, 16),
}
none_t = (ROOT / "configs/draft/bolmo-bf16-none.toml").read_text(encoding="utf-8")
p4_t = (ROOT / "configs/p4/bf16-p4.toml").read_text(encoding="utf-8")
olmo_none = (ROOT / "configs/draft/olmo-bf16-none.toml").read_text(encoding="utf-8")
olmo_k8 = (ROOT / "configs/draft/olmo-bf16-combined-k8.toml").read_text(encoding="utf-8")


def emit(name: str, text: str, quant: str, desc: str) -> None:
    text = re.sub(r'output_dir = "[^"]*"', f'output_dir = "results/skinny/{name}"', text)
    lines = text.split("\n")
    lines[0] = f"# {desc}"
    text = "\n".join(lines).replace("\n[draft]", f"\n[quantize]\n{quant}\n\n[draft]") if "[draft]" in text else text.rstrip("\n") + f"\n\n[quantize]\n{quant}\n"
    (OUT / f"{name}.toml").write_text(text, encoding="utf-8")


for v, (q, kw, kc) in BOLMO.items():
    emit(f"bolmo-{v}-none", none_t, q, f"Bolmo 1B, minimal loop, skinny kernels ({v})")
    t = re.sub(r"k_max = \d+", f"k_max = {kw}", p4_t)
    t = re.sub(r"k_cross = \d+", f"k_cross = {kc}", t)
    emit(f"bolmo-{v}-p4", t, q, f"Bolmo 1B, boundary-gated lookup drafting (P4, k_max {kw}, k_cross {kc}), skinny kernels ({v})")
olmo_q = 'target = "blocks"\nscheme = "int8_wo"\nkernel = "skinny"'
emit("olmo-int8sk-none", olmo_none, olmo_q, "OLMo 2 1B, minimal loop, int8 blocks with the skinny kernel")
emit("olmo-int8sk-k8", olmo_k8, olmo_q, "OLMo 2 1B, combined lookup K = 8, int8 blocks with the skinny kernel")
print(sorted(p.name for p in OUT.glob("*.toml")))

# piece benchmarks (tools/draft_verify_bench.py, tools/draft_global_multi_bench.py): the parallel-verification config
bench_t = (ROOT / "configs/parallel/bolmo-bf16-fused.toml").read_text(encoding="utf-8")
for v, (q, _, _) in BOLMO.items():
    emit(f"bench-{v}", bench_t, q, f"Bolmo 1B piece benchmarks (within-patch fused verify), skinny kernels ({v})")

# exactness (Stage 2.4): fp32 activations with the quantized weights (the kernels take bf16 activations, so SkinnyLinear
# dequantizes the weights to fp32 for fp32 inputs); drafted P4 against the non-drafted minimal loop of the same weights
fp32_none = (ROOT / "configs/draft/bolmo-fp32-none.toml").read_text(encoding="utf-8")
fp32_p4 = (ROOT / "configs/p4/fp32-p4.toml").read_text(encoding="utf-8")
for v in ("g8", "all4"):
    q = BOLMO[v][0]
    emit(f"fp32-{v}-none", fp32_none, q, f"Bolmo 1B, minimal loop, fp32 activations, skinny quantization ({v})")
    emit(f"fp32-{v}-p4", fp32_p4, q, f"Bolmo 1B, P4 (k_max 6, k_cross 32), fp32 activations, skinny quantization ({v})")

# OLMo 2 exactness: fp32 activations, int8 skinny blocks; truth = minimal loop, drafted = combined lookup K = 8
for nm, src in (("none", "olmo-fp32-none"), ("k8", "olmo-fp32-combined-k8")):
    t = (ROOT / f"configs/draft/{src}.toml").read_text(encoding="utf-8")
    emit(f"fp32-olmo-int8sk-{nm}", t, olmo_q, f"OLMo 2 1B, fp32 activations, int8 skinny blocks ({nm})")
