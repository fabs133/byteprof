"""Stage 0 of TASK_local_quant.md: weight bytes per output byte and predicted ms/byte.

Reads parameter shapes from the models' configs on the meta device (no weights loaded, no GPU),
applies byteprof.quant.accounting, writes results/quant/stage0.json and prints a markdown table.
Run with: python -X utf8 tools/byte_accounting.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from byteprof.quant.accounting import (  # noqa: E402
    Calibration, Variant, bolmo_components, bytes_per_output_byte, olmo_components, predict_ms,
)
from byteprof.quant.select import group_of  # noqa: E402

BOLMO = ("allenai/Bolmo-1B", "452690a8cfab1dd1ec27b74f2667632f17a21b46")
OLMO = ("allenai/OLMo-2-0425-1B", "a1847dff35000b4271fa70afc5db10fd29fedbdf")
BYTES_PER_PATCH = 3.38  # TASK_local_quant.md "Why" section (earlier profile: 3.38; minimal-loop runs: 3.3)
BYTES_PER_TOKEN = 3.39  # OLMo 2 runs, FINDINGS.md "Minimal-sync loop" section 5

# Measured bf16 minimal-loop numbers (FINDINGS.md "Minimal-sync loop", sections 3 and 5), ms per byte.
BOLMO_CAL = Calibration({"local": 0.347 + 1.131, "global": 2.310}, const_ms=3.967 - (0.347 + 1.131 + 2.310))
OLMO_MS = 2.580  # OLMo 2 compiled, minimal loop

VARIANTS = [
    Variant("bf16"),
    Variant("local int8", "local", "int8_wo"),
    Variant("local int4", "local", "int4_wo", 128),
    Variant("global int4", "global", "int4_wo", 128),
    Variant("all int4", "all", "int4_wo", 128),
    Variant("global int8", "global", "int8_wo"),
    Variant("all int8", "all", "int8_wo"),
]


def shapes(model_id: str, rev: str):
    cfg = AutoConfig.from_pretrained(model_id, revision=rev, trust_remote_code=True)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
    lin = {n: (mod.out_features, mod.in_features) for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)}
    # bf16 tensors read in full that are not weights of a counted linear: biases and norm weights
    return m, lin


def other_params(m, lin: dict, owner) -> dict[str, int]:
    out: dict[str, int] = {}
    for n, p in m.named_parameters():
        mod = n.rsplit(".", 1)[0]
        if mod in lin and n.endswith(".weight"):
            continue
        key = owner(n)
        if key:
            out[key] = out.get(key, 0) + p.numel()
    return out


def bolmo_owner(n: str):
    if n.startswith("model.local_encoder.") and "embedding" not in n and "boundary" not in n:
        return "local_encoder"
    if n.startswith("model.local_decoder."):
        return "local_decoder"
    if n.startswith("model.layers.") or n == "model.norm.weight":
        return "global_model"
    return None


def olmo_owner(n: str):
    return "layers" if n.startswith("model.layers.") else None


def main() -> None:
    mb, lb = shapes(*BOLMO)
    comps_b = bolmo_components(lb, other_params(mb, lb, bolmo_owner), BYTES_PER_PATCH, 2048, 520)
    mo, lo = shapes(*OLMO)
    comps_o = olmo_components(lo, other_params(mo, lo, olmo_owner), BYTES_PER_TOKEN, 2048, 100352)

    out: dict = {"bytes_per_patch": BYTES_PER_PATCH, "bytes_per_token": BYTES_PER_TOKEN,
                 "calibration_bolmo": {"ms_by_kind": BOLMO_CAL.ms_by_kind, "const_ms": BOLMO_CAL.const_ms},
                 "olmo_ms": OLMO_MS, "bolmo": {}, "olmo": {}}
    rows = []
    base_b = bytes_per_output_byte(comps_b)
    for v in VARIANTS:
        b = bytes_per_output_byte(comps_b, v)
        ms = predict_ms(comps_b, BOLMO_CAL, v)
        out["bolmo"][v.name] = {"bytes": b, "pred_ms": ms}
        rows.append((v.name, b, ms))
    print("### Bolmo 1B: weight MB per output byte (bf16 baseline and variants), predicted ms/byte")
    print("| variant | local enc | local dec | global | lm_head | embed | total MB | pred ms/byte | pred reduction |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, b, ms in rows:
        print(f"| {name} | {b['local_encoder']/1e6:.1f} | {b['local_decoder']/1e6:.1f} | {b['global_model']/1e6:.1f} | "
              f"{b['lm_head']/1e6:.2f} | {b['embeddings']/1e6:.3f} | {b['total']/1e6:.1f} | {ms:.3f} | "
              f"{(1 - ms / BOLMO_CAL.total_ms) * 100:.1f}% |")
    base_o = bytes_per_output_byte(comps_o)
    print()
    print("### OLMo 2 1B: weight MB per output byte")
    print("| variant | layers | lm_head | embed | total MB | pred ms/byte | pred reduction |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    # OLMo calibration: all of its measured time is weight-proportional except the constant 0.18 ms
    # (prefill + loop) that Bolmo shows; layers and head share the rest by bytes.
    const_o = BOLMO_CAL.const_ms
    for v in (Variant("bf16"), Variant("all int4", "all", "int4_wo", 128), Variant("all int8", "all", "int8_wo")):
        b = bytes_per_output_byte(comps_o, v)
        weight_ms = OLMO_MS - const_o
        ms = const_o + weight_ms * (b["total"] / base_o["total"])
        out["olmo"][v.name] = {"bytes": b, "pred_ms": ms}
        print(f"| {v.name} | {b['layers']/1e6:.1f} | {b['lm_head']/1e6:.1f} | {b['embeddings']/1e6:.3f} | "
              f"{b['total']/1e6:.1f} | {ms:.3f} | {(1 - ms / OLMO_MS) * 100:.1f}% |")

    # H14 prediction: ms saved per GB of weights quantized (stored bytes saved per output byte)
    print()
    print("### Predicted ms saved per GB of bf16 weights quantized (per output byte), int4 g128 and int8")
    for scheme in ("int8_wo", "int4_wo"):
        res = {}
        for tgt in ("local", "global"):
            v = Variant(tgt, tgt, scheme, 128)
            saved_ms = BOLMO_CAL.total_ms - predict_ms(comps_b, BOLMO_CAL, v)
            saved_gb = (base_b["total"] - bytes_per_output_byte(comps_b, v)["total"]) / 1e9
            quantized_gb_bf16 = base_b[f"kind:{tgt}"] / 1e9  # bf16 bytes of the target group read per output byte
            res[tgt] = (saved_ms, saved_gb, quantized_gb_bf16)
        print(f"- {scheme}: local saves {res['local'][0]:.3f} ms ({res['local'][1]*1e3:.1f} MB per output byte), "
              f"global {res['global'][0]:.3f} ms ({res['global'][1]*1e3:.1f} MB per output byte)")
        # per GB of *model weights* quantized (parameters, not per-output-byte reads)
        n_local = sum(l.out_features * l.in_features for c in comps_b if c.kind == "local" for l in c.linears if l.group)
        n_glob = sum(l.out_features * l.in_features for c in comps_b if c.kind == "global" for l in c.linears if l.group)
        per_gb_local = res["local"][0] / (n_local * 2 / 1e9)
        per_gb_glob = res["global"][0] / (n_glob * 2 / 1e9)
        print(f"  per GB of bf16 weights quantized (stored in the model): local {per_gb_local:.3f} ms/GB, "
              f"global {per_gb_glob:.3f} ms/GB, ratio {per_gb_local / per_gb_glob:.2f}x "
              f"(local {n_local*2/1e9:.3f} GB, global {n_glob*2/1e9:.3f} GB)")
        out[f"h14_{scheme}"] = {"local_ms_per_gb": per_gb_local, "global_ms_per_gb": per_gb_glob,
                                "ratio": per_gb_local / per_gb_glob}
    Path("results/quant").mkdir(parents=True, exist_ok=True)
    Path("results/quant/stage0.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
