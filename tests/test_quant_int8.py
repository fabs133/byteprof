import pytest
import torch
import torch.nn as nn

from byteprof.quant.triton_int8 import quantize_rowwise_int8

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton kernel)")


def test_rowwise_quantization_error_bound():
    w = torch.randn(16, 256, dtype=torch.bfloat16)
    q, s = quantize_rowwise_int8(w)
    assert q.dtype == torch.int8 and q.abs().max() <= 127 and s.dtype == torch.bfloat16
    deq = q.float() * s.float()[:, None]
    # per-row error is at most half a quantization step (plus bf16 rounding of the scale)
    assert ((deq - w.float()).abs().amax(1) <= s.float() * 0.51 + 1e-3).all()


@cuda
@pytest.mark.parametrize("n,k", [(1024, 2048), (2816, 2048), (2048, 2816), (24, 128)])
def test_gemv_matches_dequantized_reference(n, k):
    from byteprof.quant.triton_int8 import gemv_int8

    w = (torch.randn(n, k) * 0.02).to("cuda", torch.bfloat16)
    q, s = quantize_rowwise_int8(w)
    x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    ref = (x.float() @ (q.float() * s.float()[:, None]).T)
    assert (gemv_int8(x, q, s).float() - ref).abs().max() < 0.05 * ref.abs().max() + 1e-2


@cuda
def test_int8linear_single_row_and_batch_paths_agree_with_bias():
    from byteprof.quant.integrate import Int8Linear

    lin = nn.Linear(256, 64, bias=True, dtype=torch.bfloat16, device="cuda")
    m = Int8Linear(lin)
    assert m.weight.dtype == torch.bfloat16 and m.weight.numel() == 0  # what Bolmo's cache setup reads
    x = torch.randn(3, 1, 256, device="cuda", dtype=torch.bfloat16)
    batch = m(x)  # 3 rows: dequantized path
    single = torch.cat([m(x[i:i + 1]) for i in range(3)])  # 1 row each: Triton kernel
    assert batch.shape == single.shape == (3, 1, 64)
    assert (batch.float() - single.float()).abs().max() < 0.05
    assert (batch.float() - lin(x).float()).abs().max() < 0.1  # close to the bf16 layer


# --- Stage A0 (TASK_fair_ratio_floor.md): reference tests for the custom kernel -------------------

def _ref(x, q, s, bias=None):
    y = x.float() @ (q.float() * s.float()[:, None]).T  # fp32 reference on the dequantized weights
    return y if bias is None else y + bias.float()


def _close(y, ref):
    # bf16 output rounding is 2^-9 relative per element; fp32 accumulation order differs slightly
    assert y.dtype == torch.bfloat16
    assert (y.float() - ref).abs().max() <= 0.01 * ref.abs().max() + 1e-3


REAL_SHAPES = [(1024, 2048), (2048, 2048), (2816, 2048), (2048, 2816),  # local layers
               (8192, 2048), (2048, 8192)]  # global MLP


@cuda
@pytest.mark.parametrize("n,k", REAL_SHAPES)
def test_gemv_real_shapes_batch_one(n, k):
    from byteprof.quant.triton_int8 import gemv_int8

    q, s = quantize_rowwise_int8((torch.randn(n, k) * 0.02).to("cuda", torch.bfloat16))
    x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    _close(gemv_int8(x, q, s), _ref(x, q, s))


@cuda
@pytest.mark.parametrize("n,k", [(37, 1000), (1, 8), (3, 129), (1025, 2049), (16, 2047), (7, 5000)])
def test_gemv_odd_sizes_not_multiples_of_block(n, k):
    from byteprof.quant.triton_int8 import gemv_int8

    q, s = quantize_rowwise_int8((torch.randn(n, k) * 0.05).to("cuda", torch.bfloat16))
    x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16)
    _close(gemv_int8(x, q, s), _ref(x, q, s))
    for cfg in ((1, 256, 1), (8, 1024, 8)):  # block configs other than the default
        _close(gemv_int8(x, q, s, cfg), _ref(x, q, s))


@cuda
@pytest.mark.parametrize("m", [1, 2, 5, 33, 300])
@pytest.mark.parametrize("n,k,bias", [(2048, 2048, True), (2816, 2048, False), (37, 1000, True)])
def test_int8linear_all_row_counts_vs_fp32_reference(m, n, k, bias):
    from byteprof.quant.integrate import Int8Linear

    lin = nn.Linear(k, n, bias=bias, dtype=torch.bfloat16, device="cuda")
    mod = Int8Linear(lin)
    x = torch.randn(m, 1, k, device="cuda", dtype=torch.bfloat16)
    y = mod(x)
    assert y.shape == (m, 1, n)
    _close(y.reshape(m, n), _ref(x.reshape(m, k), mod.q, mod.scale, lin.bias if bias else None))


@cuda
def test_int8linear_empty_input():
    from byteprof.quant.integrate import Int8Linear

    mod = Int8Linear(nn.Linear(256, 64, bias=True, dtype=torch.bfloat16, device="cuda"))
    for shape in ((0, 256), (0, 1, 256)):
        y = mod(torch.empty(*shape, device="cuda", dtype=torch.bfloat16))
        assert y.shape == (*shape[:-1], 64) and y.numel() == 0


@cuda
def test_gemv_inside_cuda_graph_matches_eager():
    from byteprof.quant.triton_int8 import gemv_int8

    q, s = quantize_rowwise_int8((torch.randn(2048, 2048) * 0.02).to("cuda", torch.bfloat16))
    x = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    gemv_int8(x, q, s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = gemv_int8(x, q, s)
    x.copy_(torch.randn_like(x))
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, gemv_int8(x, q, s))
