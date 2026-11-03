"""Quantize a loaded model's selected linears in place (imported only when target != "none").

int4_wo: torchao `int4_weight_only(group_size)` (tinygemm, `torch._weight_int4pack_mm`, dequantizes
inside the kernel). int8_wo: `Int8Linear` below, per-row symmetric int8 with the fused Triton GEMV
for single-row inputs (torchao 0.9's own int8 weight-only path dequantizes in a separate step and is
slower than bf16 here, see FINDINGS.md Stage 1). Inputs with more than one row (prefill, bits-per-byte
evaluation) use the same int8 weights dequantized to bf16 for that call.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .select import group_of, olmo_should_quantize, should_quantize
from .triton_int8 import gemv_int8, quantize_rowwise_int8


class Int8Linear(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        q, scale = quantize_rowwise_int8(lin.weight.data)
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        # Bolmo's remote code reads `out_projection.weight.dtype` / `in_projection.weight.dtype` when it
        # sets up its cache; an empty bf16 parameter keeps that working without holding the bf16 weights.
        self.weight = nn.Parameter(torch.empty(0, dtype=lin.weight.dtype, device=lin.weight.device),
                                   requires_grad=False)
        self.bias = lin.bias

    def dequantized(self, dtype: torch.dtype) -> torch.Tensor:
        return self.q.to(dtype) * self.scale.to(dtype)[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, self.in_features)
        if x2.shape[0] == 1 and x2.is_cuda:
            y = gemv_int8(x2, self.q, self.scale)
            if self.bias is not None:
                y = y + self.bias
        else:
            y = F.linear(x2, self.dequantized(x.dtype), self.bias)
        return y.reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8 rowwise"


def _allow_empty_input(lin: nn.Linear) -> None:
    """The int4 kernel cannot launch for zero rows. Bolmo applies `out_projection` to an empty
    (0, 1, D) tensor on every non-boundary step (and in prefill), which a bf16 linear handles."""
    orig = lin.forward
    n_out = lin.out_features

    def forward(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_empty(*x.shape[:-1], n_out)
        return orig(x)

    lin.forward = forward


def apply_quantization(model: nn.Module, target: str, scheme: str, group_size: int = 128,
                       local_scheme: str | None = None, global_scheme: str | None = None,
                       kernel: str = "default") -> dict[str, Any]:
    """Quantize every linear selected by `should_quantize(name, target)`; returns statistics.

    `local_scheme` / `global_scheme` override `scheme` for that group of Bolmo (e.g. local int4 +
    global int8).

    `kernel = "skinny"` swaps every selected linear for a `SkinnyLinear` (byteprof.optim.skinny_linear: one family of
    Triton kernels for M = 1..64 rows; schemes bf16_wo, int8_wo, int4_wo with the kernel's own int4 format); the default
    torchao / Int8Linear paths stay as they are."""
    bolmo = hasattr(model.model, "local_encoder")
    pick = should_quantize if bolmo else olmo_should_quantize
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and pick(n, target)]
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    def scheme_of(n: str) -> str:
        return (local_scheme if group_of(n) == "local" else global_scheme if group_of(n) == "global" else None) or scheme

    by_scheme: dict[str, list[str]] = {}
    for n in names:
        by_scheme.setdefault(scheme_of(n), []).append(n)
    for sch, group_names in by_scheme.items():
        if kernel == "skinny":
            from ..optim.skinny_linear import SkinnyLinear

            for n in group_names:
                parent_name, _, child = n.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child, SkinnyLinear(getattr(parent, child), sch))
        elif sch == "int4_wo":
            from torchao.quantization import int4_weight_only, quantize_

            selected = set(group_names)
            quantize_(model, int4_weight_only(group_size=group_size), filter_fn=lambda m, fqn: fqn in selected)
            for n in group_names:
                _allow_empty_input(model.get_submodule(n))
        elif sch == "int8_wo":
            for n in group_names:
                parent_name, _, child = n.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child, Int8Linear(getattr(parent, child)))
        else:
            raise ValueError(f"unknown scheme {sch!r}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    stats = {"target": target, "scheme": scheme, "local_scheme": local_scheme, "global_scheme": global_scheme,
             "group_size": group_size if "int4_wo" in by_scheme else None,
             "n_linears": len(names), "gpu_mem_before_mib": before / 2**20, "gpu_mem_after_mib": after / 2**20}
    if kernel != "default":
        stats["kernel"] = kernel
    if "int4_wo" in by_scheme and kernel == "default":
        import torchao

        stats["torchao"] = torchao.__version__
    return stats
