"""Drop-in replacement for an `nn.Linear` that runs the skinny kernels (`byteprof.optim.skinny`).

Schemes: bf16_wo (the original bf16 weights), int8_wo (per-row symmetric int8, the same quantization as
`byteprof.quant.integrate.Int8Linear`), int4_wo (the kernel's own group-128 format, `skinny.quantize_int4`).
Inputs with 1..MAX_M rows (decode steps, drafts, global passes at P patches, local verification at K + 1 rows)
use the Triton kernel; empty inputs return an empty tensor; larger inputs (prefill, bits-per-byte evaluation) and
non-bf16 activations (the fp32 exactness runs) use the same weights dequantized to the activation dtype for that call,
as `Int8Linear` does.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import skinny


class SkinnyLinear(nn.Module):
    def __init__(self, lin: nn.Linear, scheme: str):
        super().__init__()
        if scheme not in ("bf16_wo", "int8_wo", "int4_wo"):
            raise ValueError(f"unknown scheme {scheme!r}")
        self.scheme = scheme
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.bias = lin.bias
        w = lin.weight.data
        if scheme == "bf16_wo":
            self.weight = lin.weight  # kept as is (Bolmo's cache setup reads weight.dtype)
            self.scale = self.zero = None
            return
        # Bolmo's remote code reads `weight.dtype`; an empty parameter keeps that working without the bf16 weights.
        self.weight = nn.Parameter(torch.empty(0, dtype=w.dtype, device=w.device), requires_grad=False)
        if scheme == "int8_wo":
            from ..quant.triton_int8 import quantize_rowwise_int8

            q, scale = quantize_rowwise_int8(w)
            self.register_buffer("q", q)
            self.register_buffer("scale", scale)
            self.zero = None
        else:
            packed, scale, zero = skinny.quantize_int4(w)
            self.register_buffer("q", packed)
            self.register_buffer("scale", scale)
            self.register_buffer("zero", zero)

    def _kind(self) -> str:
        return {"bf16_wo": "bf16", "int8_wo": "int8", "int4_wo": "int4"}[self.scheme]

    def dequantized(self, dtype: torch.dtype) -> torch.Tensor:
        if self.scheme == "bf16_wo":
            return self.weight
        return skinny.dequantize(self._kind(), self.q, self.scale, self.zero).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, self.in_features)
        if x2.shape[0] == 0:
            return x.new_empty(*x.shape[:-1], self.out_features)
        if x2.is_cuda and x2.dtype == torch.bfloat16 and x2.shape[0] <= skinny.MAX_M:
            if not x2.is_contiguous():
                x2 = x2.contiguous()
            w = self.weight if self.scheme == "bf16_wo" else self.q
            y = skinny.skinny_mm(x2, w, self.scale, self.zero)
            if self.bias is not None:
                y = y + self.bias
        else:
            y = F.linear(x2, self.dequantized(x.dtype), self.bias)
        return y.reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, {self.scheme}, skinny kernel"
