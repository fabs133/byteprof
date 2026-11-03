"""Skinny weight-only matmul kernels (M = 1..33 rows), Triton (TASK_skinny_kernels.md).

y[M, N] = x[M, K] @ W[N, K]^T with bf16 activations, fp32 accumulation, bf16 output, and the weights
stored as bf16, int8 per output channel, or int4 in groups of 128 along K. Each program owns BN
output channels and streams their weights exactly once; all M activation rows are applied to every
weight tile (swap-AB: the weight tile is the MMA's M operand, the few activation rows its N operand),
so the extra rows ride on idle tensor-core throughput instead of re-reading the weights.

Weight formats
  bf16  W: [N, K] bf16
  int8  W: [N, K] int8, scale [N] (bf16):  w ~ q * scale[n]                   (byteprof.quant.triton_int8)
  int4  W: uint8 [ceil(N/R), K/128, R, 64], scale [K/128, N] (bf16) and zero [K/128, N] (fp16):
        w ~ scale[n, g] * (q - zero[n, g]), q in 0..15 (asymmetric group quantization with a float zero point, the
        same family as torchao's int4 groups; scale and zero cost 4 bytes per group, as tinygemm's scales_and_zeros).
        Own packing (`pack_int4`): inside each group of 128 along K, byte j (0..63) holds code j in its low nibble
        and code j + 64 in its high nibble (no nibble interleave in the kernel); the R rows (R = 32) of one group form
        one contiguous R * 64 byte block, and the blocks of a row tile follow each other along K, so each program
        streams one contiguous region (a plain [N, K/2] layout reads 64-byte row segments and ran at ~half the
        bandwidth). K must be a multiple of 128 for int4.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import triton
import triton.language as tl

GROUP = 128
MAX_M = 64  # rows per launch; more rows are processed in chunks of MAX_M (the weights are re-read per chunk)


@triton.jit
def _skinny_kernel(x_ptr, w_ptr, s_ptr, z_ptr, y_ptr, M, N, K,
                   MODE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   STAGES: tl.constexpr, SPLIT: tl.constexpr):
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    rn = pid * BN + tl.arange(0, BN)
    nmask = rn < N
    rm = tl.arange(0, BM)
    mmask = rm < M
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    if MODE == 3:
        # int4 (tile-major format, see module docstring). The nibble q goes into the low mantissa bits of fp32 128.0
        # (0x4300 | q in the top half-word = 128 + q, exact); subtracting 128 + c per row gives q - c, rounded once to
        # bf16 for the MMA; the group scale is applied to the [BN, BM] partial: y_g = scale * sum_k (q - c) x.
        rh = tl.arange(0, 64)
        NG = K // 128
        base = w_ptr + pid * (NG * BN * 64) + tl.arange(0, BN)[:, None] * 64 + rh[None, :]
        for g in tl.range(sid * NG // SPLIT, (sid + 1) * NG // SPLIT, num_stages=STAGES):
            packed = tl.load(base + g * (BN * 64), eviction_policy="evict_first").to(tl.int32)
            sc = tl.load(s_ptr + g * N + rn, mask=nmask, other=0.0).to(tl.float32)
            cf = tl.load(z_ptr + g * N + rn, mask=nmask, other=0.0).to(tl.float32) + 128.0
            wlo = ((((packed & 0xF) | 0x4300) << 16).to(tl.float32, bitcast=True) - cf[:, None]).to(tl.bfloat16)
            whi = (((((packed >> 4) & 0xF) | 0x4300) << 16).to(tl.float32, bitcast=True) - cf[:, None]).to(tl.bfloat16)
            xlo = tl.load(x_ptr + rm[None, :] * K + (g * 128 + rh)[:, None], mask=mmask[None, :], other=0.0)
            xhi = tl.load(x_ptr + rm[None, :] * K + (g * 128 + 64 + rh)[:, None], mask=mmask[None, :], other=0.0)
            part = tl.dot(wlo, xlo)
            part = tl.dot(whi, xhi, part)
            acc += sc[:, None] * part
    else:
        NB = (K + BK - 1) // BK
        for kb in tl.range(sid * NB // SPLIT, (sid + 1) * NB // SPLIT, num_stages=STAGES):
            rk = kb * BK + tl.arange(0, BK)
            kmask = rk < K
            w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=nmask[:, None] & kmask[None, :], other=0,
                        eviction_policy="evict_first")
            x = tl.load(x_ptr + rm[None, :] * K + rk[:, None], mask=mmask[None, :] & kmask[:, None], other=0.0)
            acc = tl.dot(w.to(tl.bfloat16), x, acc)
        if MODE == 1:
            sc = tl.load(s_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
            acc = acc * sc[:, None]
    if SPLIT == 1:
        tl.store(y_ptr + rm[None, :] * N + rn[:, None], acc.to(y_ptr.dtype.element_ty),
                 mask=nmask[:, None] & mmask[None, :])
    else:  # fp32 partial sums, reduced by _split_reduce
        tl.store(y_ptr + sid * (M * N) + rm[None, :] * N + rn[:, None], acc, mask=nmask[:, None] & mmask[None, :])


@triton.jit
def _split_reduce(ws_ptr, y_ptr, total, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.static_range(SPLIT):
        acc += tl.load(ws_ptr + i * total + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, acc.to(y_ptr.dtype.element_ty), mask=mask)


# (BN, BK, num_warps, num_stages[, split_k]) per (kind, N, K, BM); BM is chosen from M. Filled by tools/skinny_tune.py.
_CFG: dict[tuple[str, int, int, int], tuple[int, ...]] = {}
_CFG_FILE = Path(__file__).with_name("skinny_cfg.json")
if _CFG_FILE.exists():
    for _key, _v in json.loads(_CFG_FILE.read_text(encoding="utf-8")).items():
        _kind, _n, _k, _bm = _key.split("|")
        _CFG[(_kind, int(_n), int(_k), int(_bm))] = tuple(_v)  # type: ignore[assignment]
_DEFAULT = {"bf16": (32, 128, 4, 3), "int8": (32, 256, 4, 3), "int4": (32, 128, 4, 3)}


def _bm(m: int) -> int:
    return 16 if m <= 16 else 32 if m <= 32 else 64


def get_cfg(kind: str, n: int, k: int, bm: int) -> tuple[int, ...]:
    return _CFG.get((kind, n, k, bm), _CFG.get((kind, n, k, 0), _DEFAULT[kind]))  # type: ignore[call-overload]


def _launch(kind: str, x, w, scale, zero, y, cfg=None):
    m, k = x.shape
    bm = _bm(m)
    if kind == "int4":
        n = scale.shape[1]
        _, bk, nw, st, split = _full(cfg or get_cfg(kind, n, k, bm))
        bn, grid = w.shape[2], w.shape[0]  # one program per row tile of the packed format
        split = min(split, k // GROUP)
    else:
        n = w.shape[0]
        bn, bk, nw, st, split = _full(cfg or get_cfg(kind, n, k, bm))
        grid = triton.cdiv(n, bn)
        split = min(split, triton.cdiv(k, bk))
    mode = {"bf16": 0, "int8": 1, "int4": 3}[kind]
    out = y if split == 1 else torch.empty(split, m, n, dtype=torch.float32, device=x.device)
    _skinny_kernel[(grid, split)](x, w, scale if scale is not None else w, zero if zero is not None else w,
                                  out, m, n, k, MODE=mode, BM=bm, BN=bn, BK=bk, STAGES=st, SPLIT=split, num_warps=nw)
    if split > 1:
        _split_reduce[(triton.cdiv(m * n, 1024),)](out, y, m * n, SPLIT=split, BLOCK=1024)


def _full(cfg):
    return tuple(cfg) if len(cfg) == 5 else (*cfg, 1)


def skinny_mm(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor | None = None,
              zero: torch.Tensor | None = None, out: torch.Tensor | None = None,
              cfg: tuple[int, ...] | None = None) -> torch.Tensor:
    """x: [M, K] bf16 (contiguous); w: bf16 [N, K] | int8 [N, K] + scale [N] | int4 tile-major uint8 + scale, zero [K/128, N]."""
    if x.dim() != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous [M, K] tensor")
    m, k = x.shape
    if w.dtype == torch.bfloat16:
        kind, n = "bf16", w.shape[0]
        if w.shape[1] != k:
            raise ValueError("K mismatch")
    elif w.dtype == torch.int8:
        kind, n = "int8", w.shape[0]
        if scale is None or w.shape[1] != k:
            raise ValueError("int8 needs scale [N] and K to match")
    elif w.dtype == torch.uint8:
        kind = "int4"
        if scale is None or zero is None or w.dim() != 4 or k % GROUP or w.shape[1] * GROUP != k or scale.shape[0] != w.shape[1]:
            raise ValueError("int4 needs packed [N/R, K/128, R, 64], scale/zero [K/128, N] and K % 128 == 0")
        n = scale.shape[1]
    else:
        raise ValueError(f"unsupported weight dtype {w.dtype}")
    y = out if out is not None else torch.empty(m, n, dtype=x.dtype, device=x.device)
    if m == 0 or n == 0 or k == 0:
        if k == 0:
            y.zero_()
        return y
    for r0 in range(0, m, MAX_M):
        xs = x[r0:r0 + MAX_M]
        _launch(kind, xs, w, scale, zero, y[r0:r0 + MAX_M], cfg)
    return y


# ---- weight preparation ----------------------------------------------------------------------------


def quantize_int4(w: torch.Tensor, tile_rows: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric group-wise int4 of w [N, K] -> (packed uint8 [ceil(N/R), K/128, R, 64], scale [K/128, N] in w.dtype,
    zero [K/128, N] fp16): scale = (max - min) / 15, zero = -min / scale, q = round(w / scale + zero) clamped to 0..15,
    so scale * (q - zero) spans [min, max] up to rounding."""
    n, k = w.shape
    if k % GROUP:
        raise ValueError("int4 needs K % 128 == 0")
    wg = w.float().reshape(n, k // GROUP, GROUP)
    mn, mx = wg.amin(-1), wg.amax(-1)
    scale = ((mx - mn) / 15.0).clamp_min(1e-6).to(w.dtype)
    zero = (-mn / scale.float()).to(torch.float16)
    q = torch.round(wg / scale.float()[..., None] + zero.float()[..., None]).clamp_(0, 15).to(torch.uint8)
    return pack_int4(q.reshape(n, k), tile_rows), scale.t().contiguous(), zero.t().contiguous()


def pack_int4(q: torch.Tensor, tile_rows: int = 32) -> torch.Tensor:
    """q: uint8 codes 0..15 [N, K] -> [ceil(N/R), K/128, R, 64]. Inside a group of 128 along K, byte j holds code j in
    its low nibble and code j + 64 in its high nibble; R rows of one group form one contiguous R * 64 byte block
    (padding rows are zero), the blocks of a row tile follow each other along K."""
    n, k = q.shape
    r = tile_rows
    npad = -(-n // r) * r
    if npad != n:
        q = torch.cat([q, q.new_zeros(npad - n, k)])
    g = q.reshape(npad // r, r, k // GROUP, 2, GROUP // 2)
    packed = g[:, :, :, 0] | (g[:, :, :, 1] << 4)  # [T, R, G, 64]
    return packed.permute(0, 2, 1, 3).contiguous()


def unpack_int4(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of pack_int4 -> codes [N, K]."""
    t, ng, r, _ = packed.shape
    p = packed.permute(0, 2, 1, 3)  # [T, R, G, 64]
    q = torch.stack([p & 0xF, p >> 4], dim=3)  # [T, R, G, 2, 64]
    return q.reshape(t * r, ng * GROUP)[:n]


def dequantize(kind: str, w: torch.Tensor, scale=None, zero=None) -> torch.Tensor:
    """fp32 reference weights [N, K]."""
    if kind == "bf16":
        return w.float()
    if kind == "int8":
        return w.float() * scale.float()[:, None]
    ng, n = scale.shape
    q = unpack_int4(w, n).float().reshape(n, ng, GROUP) - zero.float().t()[..., None]
    return (q * scale.float().t()[..., None]).reshape(n, ng * GROUP)
