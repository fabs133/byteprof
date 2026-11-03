"""Skinny weight-only matmul kernels (TASK_skinny_kernels.md): format round trips (CPU) and GPU reference checks of
`byteprof.optim.skinny` against an fp32 dequantize-then-matmul reference for every M = 1..33 and all real layer shapes.

Documented tolerance (H59): componentwise error  max_ij |y - ref|_ij / (|x| |W|^T)_ij  <= TOL = 4e-3, i.e. relative to the sum of
the absolute terms of each output (so outputs that nearly cancel do not dominate; a first version normalised by max|ref| and
failed on a 1 x 128 int4 layer whose outputs cancel). Sources: the bf16 output rounding (2^-9), bf16 operands for the MMA (the
int4 dequantized value q - c is rounded once to bf16) and fp32 accumulation in a different order than the reference. The
measured maxima are below it (see FINDINGS.md)."""

import pytest
import torch

from byteprof.optim import skinny as S
from byteprof.quant.triton_int8 import quantize_rowwise_int8

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
TOL = 4e-3  # about 2^-8: output rounding (2^-9) plus the int4 dequantized-operand rounding (2^-9), each relative to the sum of absolute terms
# real shapes (out, in) of Bolmo's global / local layers and OLMo 2's blocks, plus odd sizes
REAL = [(2048, 2048), (8192, 2048), (2048, 8192), (1024, 2048), (2816, 2048), (2048, 2816)]
ODD = [(37, 256), (1, 128), (50, 384), (129, 640)]


def make(kind, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    if kind == "bf16":
        return w, (w,)
    if kind == "int8":
        q, s = quantize_rowwise_int8(w)
        return w, (q, s)
    return w, S.quantize_int4(w)


def nerr(y, x, ref_w):
    """Componentwise error max_ij |y - ref|_ij / (|x| |W|^T)_ij against the fp32 dequantize-then-matmul reference. Normalising by
    the sum of absolute terms (not by |ref|) keeps outputs that nearly cancel from dominating: the operand and output rounding
    errors are bounded by it (about 2 * 2^-9), whatever the cancellation."""
    ref = x.float() @ ref_w.T
    scale = (x.float().abs() @ ref_w.abs().T).clamp(min=1e-30)
    return float(((y.float() - ref).abs() / scale).max())


# ---- formats (CPU) ----------------------------------------------------------------------------------


@pytest.mark.parametrize("n,rows", [(32, 32), (37, 32), (5, 16), (64, 16)])
def test_int4_pack_round_trip(n, rows):
    q = torch.randint(0, 16, (n, 384), dtype=torch.uint8)
    packed = S.pack_int4(q, rows)
    assert packed.shape == (-(-n // rows), 3, rows, 64) and packed.dtype == torch.uint8
    assert torch.equal(S.unpack_int4(packed, n), q)
    # the documented layout: byte j of a group holds code j (low nibble) and code j + 64 (high nibble)
    assert int(packed[0, 1, 0, 5]) == int(q[0, 128 + 5]) | (int(q[0, 128 + 64 + 5]) << 4)


def test_int4_quantization_error_is_at_most_half_a_step():
    w = (torch.randn(24, 256) * 0.05).to(torch.bfloat16)
    packed, scale, zero = S.quantize_int4(w)
    assert scale.shape == zero.shape == (2, 24) and zero.dtype == torch.float16 and scale.dtype == torch.bfloat16
    deq = S.dequantize("int4", packed, scale, zero)
    step = scale.float().t().repeat_interleave(128, dim=1)
    assert ((deq - w.float()).abs() <= 0.55 * step + 1e-3).all()
    assert S.unpack_int4(packed, 24).max() <= 15


def test_int4_rejects_k_not_multiple_of_group():
    with pytest.raises(ValueError):
        S.quantize_int4(torch.zeros(4, 100, dtype=torch.bfloat16))


# ---- GPU --------------------------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize("kind", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("n,k", REAL + ODD)
def test_matches_fp32_reference_for_every_m(kind, n, k):
    if kind == "int4" and k % 128:
        pytest.skip("int4 needs K % 128 == 0")
    _, args = make(kind, n, k)
    ref_w = S.dequantize(kind, *args)
    worst = 0.0
    for m in range(1, 34):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        y = S.skinny_mm(x, *args)
        assert y.shape == (m, n) and y.dtype == torch.bfloat16
        worst = max(worst, nerr(y, x, ref_w))
    assert worst < TOL, worst


@cuda
@pytest.mark.parametrize("kind", ["bf16", "int8", "int4"])
def test_empty_input_and_many_rows(kind):
    _, args = make(kind, 64, 256)
    ref_w = S.dequantize(kind, *args)
    assert S.skinny_mm(torch.empty(0, 256, device="cuda", dtype=torch.bfloat16), *args).shape == (0, 64)
    x = torch.randn(150, 256, device="cuda", dtype=torch.bfloat16)  # more than MAX_M rows: chunked
    assert nerr(S.skinny_mm(x, *args), x, ref_w) < TOL


@cuda
@pytest.mark.parametrize("kind,cfg", [("bf16", (32, 128, 4, 3, 4)), ("int8", (16, 256, 4, 3, 3)), ("int4", (32, 128, 4, 3, 5)),
                                      ("int4", (32, 128, 2, 2, 100)), ("bf16", (64, 64, 2, 2, 1))])
def test_split_k_and_other_configs_match(kind, cfg):
    _, args = make(kind, 192, 1024)
    ref_w = S.dequantize(kind, *args)
    for m in (1, 5, 17, 33):
        x = torch.randn(m, 1024, device="cuda", dtype=torch.bfloat16)
        assert nerr(S.skinny_mm(x, *args, cfg=cfg), x, ref_w) < TOL


@cuda
@pytest.mark.parametrize("kind", ["bf16", "int8", "int4"])
def test_rows_do_not_depend_on_the_other_rows(kind):
    """Output row i must not change with the number of rows in the same BM bucket (M-invariance within a tile
    shape: every row has its own accumulator and the same K order)."""
    _, args = make(kind, 2048, 2048)
    x = torch.randn(16, 2048, device="cuda", dtype=torch.bfloat16)
    full = S.skinny_mm(x, *args)
    for m in (1, 3, 9):
        assert torch.equal(S.skinny_mm(x[:m].contiguous(), *args), full[:m])


@cuda
def test_input_validation():
    w = torch.zeros(8, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        S.skinny_mm(torch.zeros(2, 64, device="cuda", dtype=torch.bfloat16), w)  # K mismatch
    with pytest.raises(ValueError):
        S.skinny_mm(torch.zeros(2, 128, device="cuda", dtype=torch.bfloat16)[:, ::2], w)  # not contiguous
    with pytest.raises(ValueError):
        S.skinny_mm(torch.zeros(2, 128, device="cuda", dtype=torch.bfloat16), torch.zeros(8, 128, device="cuda", dtype=torch.int8))  # no scale


@cuda
@pytest.mark.parametrize("kind", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("m", [1, 3, 17])
def test_kernel_writes_only_the_output_block(kind, m):
    """Rows beyond M of the row block and columns beyond N must not be written: the output sits in the middle of a sentinel buffer."""
    n, k = 70, 384  # N not a multiple of the tile, K not a multiple of the K block
    _, args = make(kind, n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    flat = torch.full((m * n + 4096,), 777.0, device="cuda", dtype=torch.bfloat16)
    out = flat[:m * n].view(m, n)
    S.skinny_mm(x, *args, out=out)
    assert torch.equal(flat[m * n:], torch.full_like(flat[m * n:], 777.0))
    assert not (out == 777.0).any()
    ref_w = S.dequantize(kind, *args)
    assert nerr(out, x, ref_w) < TOL
