"""`quantize.kernel = "skinny"` (TASK_skinny_kernels.md): config validation, the default paths staying unchanged, and the
SkinnyLinear swap on a toy OLMo-shaped model."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from byteprof.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parent.parent
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _write(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    return p


# ---- config ------------------------------------------------------------------------------------------


def test_kernel_key_default_and_validation(tmp_path):
    cfg = load_config(_write(tmp_path, '[quantize]\ntarget = "all"\nscheme = "int8_wo"\n'))
    assert "kernel" not in cfg.quantize  # absent = default kernels, same table as before
    ok = load_config(_write(tmp_path, '[quantize]\ntarget = "all"\nscheme = "bf16_wo"\nglobal_scheme = "int8_wo"\nkernel = "skinny"\n'))
    assert ok.quantize["kernel"] == "skinny"
    for bad in ('target = "all"\nscheme = "int8_wo"\nkernel = "fast"',
                'target = "all"\nscheme = "bf16_wo"',  # bf16_wo only exists for the skinny kernel
                'target = "all"\nscheme = "bf16_wo"\nkernel = "default"',
                'target = "all"\nscheme = "int4_wo"\nkernel = "skinny"\ngroup_size = 64'):
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, f"[quantize]\n{bad}\n"))


def test_default_kernel_never_imports_the_skinny_code():
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
        "from byteprof.config import load_config\n"
        "from byteprof import runner, cli\n"
        "for ov in (None, {'quantize': {'target': 'none', 'scheme': 'int4_wo'}}, {'quantize': {'target': 'none', 'kernel': 'skinny'}}):\n"
        "    cfg = load_config(None, ov)\n"
        "bad = [m for m in sys.modules if 'skinny' in m or m.startswith('byteprof.optim') or m.startswith('byteprof.quant')]\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_skinny_code_is_imported_only_behind_the_kernel_guard():
    src = (ROOT / "byteprof" / "quant" / "integrate.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert src.count("skinny_linear") == 2  # the guarded import and the docstring
    guard = src.index('if kernel == "skinny":')
    assert guard < src.index("from ..optim.skinny_linear import SkinnyLinear") < guard + 100
    assert "elif sch == \"int4_wo\":" in src and "elif sch == \"int8_wo\":" in src  # default branches intact
    for f in ("runner.py", "config.py", "cli.py"):
        text = (ROOT / "byteprof" / f).read_text(encoding="utf-8")
        imports = "\n".join(l for l in text.splitlines() if l.lstrip().startswith(("import ", "from ")))
        assert "skinny" not in imports, f


# ---- the swap on a toy model -------------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = nn.ModuleDict({n: nn.Linear(d, d, bias=False) for n in ("q_proj", "k_proj", "v_proj", "o_proj")})
        self.mlp = nn.ModuleDict({"gate_proj": nn.Linear(d, 2 * d, bias=False), "up_proj": nn.Linear(d, 2 * d, bias=False),
                                  "down_proj": nn.Linear(2 * d, d, bias=False)})


class _Toy(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(d) for _ in range(2)])
        self.lm_head = nn.Linear(d, 64, bias=False)


def _toy(seed=0):
    torch.manual_seed(seed)
    return _Toy().to("cuda", torch.bfloat16)


@cuda
@pytest.mark.parametrize("scheme", ["bf16_wo", "int8_wo", "int4_wo"])
def test_apply_quantization_with_the_skinny_kernel_swaps_selected_linears(scheme):
    from byteprof.optim.skinny_linear import SkinnyLinear
    from byteprof.quant.integrate import apply_quantization

    m = _toy()
    stats = apply_quantization(m, "blocks", scheme, 128, None, None, "skinny")
    assert stats["kernel"] == "skinny" and stats["n_linears"] == 14
    swapped = [x for x in m.modules() if isinstance(x, SkinnyLinear)]
    assert len(swapped) == 14 and isinstance(m.lm_head, nn.Linear)  # head stays bf16
    x = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16)
    ref = _toy()
    for rows in (1, 5, 100):  # kernel, kernel, dequantize fallback
        xr = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        a = m.model.layers[0].self_attn["q_proj"](xr).float()
        b = ref.model.layers[0].self_attn["q_proj"](xr).float()
        assert a.shape == b.shape == (rows, 256)
        tol = {"bf16_wo": 1e-2, "int8_wo": 0.1, "int4_wo": 0.5}[scheme]  # vs the bf16 layer: quantization error only
        assert (a - b).abs().max() < tol * b.abs().max()
    assert m.model.layers[0].self_attn["q_proj"](x.new_empty(0, 256)).shape == (0, 256)
    assert m.model.layers[0].self_attn["q_proj"].weight.dtype == torch.bfloat16  # what Bolmo's cache setup reads


@cuda
def test_default_kernel_still_builds_the_old_modules_and_stats():
    from byteprof.quant.integrate import Int8Linear, apply_quantization

    m = _toy()
    stats = apply_quantization(m, "blocks", "int8_wo")
    assert "kernel" not in stats  # the default path's statistics are unchanged
    assert sum(isinstance(x, Int8Linear) for x in m.modules()) == 14
    with pytest.raises(ValueError):
        apply_quantization(_toy(), "blocks", "bf16_wo")  # skinny-only scheme on the default path


@cuda
def test_skinny_int8_uses_the_same_quantized_weights_as_the_default_int8():
    from byteprof.optim.skinny_linear import SkinnyLinear
    from byteprof.quant.integrate import Int8Linear

    lin = nn.Linear(512, 128, bias=False, dtype=torch.bfloat16, device="cuda")
    a, b = Int8Linear(lin), SkinnyLinear(lin, "int8_wo")
    assert torch.equal(a.q, b.q) and torch.equal(a.scale, b.scale)
    x = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
    assert (a(x).float() - b(x).float()).abs().max() < 0.02 * a(x).float().abs().max()


@cuda
def test_skinny_linear_inside_a_cuda_graph_with_bias():
    from byteprof.optim.skinny_linear import SkinnyLinear

    lin = nn.Linear(256, 96, bias=True, dtype=torch.bfloat16, device="cuda")
    for scheme in ("bf16_wo", "int8_wo", "int4_wo"):
        m = SkinnyLinear(lin, scheme)
        x = torch.randn(3, 1, 256, device="cuda", dtype=torch.bfloat16)
        eager = m(x)
        with torch.inference_mode():
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                m(x)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y = m(x)
            x.copy_(torch.randn_like(x))
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(y, m(x))
        assert eager.shape == (3, 1, 96)


@cuda
@pytest.mark.parametrize("scheme", ["int8_wo", "int4_wo"])
def test_fp32_activations_use_the_dequantized_weights(scheme):
    from byteprof.optim.skinny_linear import SkinnyLinear

    lin = nn.Linear(256, 64, bias=False, dtype=torch.float32, device="cuda")
    m = SkinnyLinear(lin, scheme)
    x = torch.randn(3, 256, device="cuda", dtype=torch.float32)
    y = m(x)
    assert y.dtype == torch.float32
    assert torch.allclose(y, x @ m.dequantized(torch.float32).T, atol=1e-5, rtol=1e-5)
    assert (y - lin(x)).abs().max() < (0.1 if scheme == "int8_wo" else 0.5) * lin(x).abs().max()
    # cuBLAS fp32 rounds an M = 1 and an M = 3 call differently in the last bits (gemv vs gemm), so only agreement to fp32 resolution
    assert torch.allclose(m(x[:1]), y[:1], atol=1e-5, rtol=1e-5)
