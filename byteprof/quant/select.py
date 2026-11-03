"""Which linear layers a `[quantize]` target selects (pure: module names in, bool out).

Never selected, whatever the target: embedding tables (one row is read per step), `lm_head`
(520 outputs, 2 MB), the boundary predictor (prefill only, and it decides patching),
and the two 16-row mLSTM gate projections `igate_preact` / `fgate_preact` (65 KB each; they
set the recurrence's input and forget gates, and their output size is not worth a packed kernel).
"""

from __future__ import annotations

import re

BOLMO_TARGETS = ("none", "local", "global", "all")
OLMO_TARGETS = ("blocks", "blocks+head")  # OLMo 2: transformer layers only / layers + lm_head
VALID_TARGETS = BOLMO_TARGETS + OLMO_TARGETS
VALID_SCHEMES = ("int8_wo", "int4_wo")

_LOCAL = re.compile(
    r"^model\.local_(encoder|decoder)\.layers\.\d+\."
    r"(xlstm\.(q|k|v|ogate_preact|out_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
    r"|^model\.local_encoder\.out_projection$"
    r"|^model\.local_decoder\.in_projection$"
)
# Bolmo's global transformer and OLMo 2's transformer layers share this layout.
_GLOBAL = re.compile(r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate_proj|up_proj|down_proj))$")


def group_of(module_name: str) -> str | None:
    """"local", "global" or None for a linear that is never quantized."""
    if _LOCAL.match(module_name):
        return "local"
    if _GLOBAL.match(module_name):
        return "global"
    return None


def should_quantize(module_name: str, target: str) -> bool:
    """Bolmo: none | local | global | all."""
    if target not in BOLMO_TARGETS:
        raise ValueError(f"Bolmo target must be one of {list(BOLMO_TARGETS)}, got {target!r}")
    g = group_of(module_name)
    return g is not None and (target == "all" or target == g)


def olmo_group_of(module_name: str) -> str | None:
    """"blocks" for the linears of OLMo 2's transformer layers, "head" for lm_head, else None.

    Embedding tables, norms and everything else are never selected."""
    if _GLOBAL.match(module_name):
        return "blocks"
    return "head" if module_name == "lm_head" else None


def olmo_should_quantize(module_name: str, target: str) -> bool:
    """OLMo 2: none | blocks | blocks+head."""
    if target not in ("none",) + OLMO_TARGETS:
        raise ValueError(f"OLMo target must be one of {['none', *OLMO_TARGETS]}, got {target!r}")
    g = olmo_group_of(module_name)
    return g is not None and (g == "blocks" and target in OLMO_TARGETS or g == "head" and target == "blocks+head")


def selects(target: str, group: str | None) -> bool:
    """Does `target` quantize a linear of this group (groups: local, global, blocks, head)?"""
    if group is None or target == "none":
        return False
    return {"local": group == "local", "global": group == "global", "all": group in ("local", "global"),
            "blocks": group == "blocks", "blocks+head": group in ("blocks", "head")}[target]
