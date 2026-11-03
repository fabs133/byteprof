"""Patch-aligned task: how much of P4's predicted gain survives if the multi-patch round costs more than measured.

Usage: python tools/aligned_sensitivity.py -> results/aligned/sensitivity.md
Variants of the cost of a crossing round (a round whose draft is verified with the crossing rule): the measured B(P) slope
over P doubled and tripled, and a flat extra cost per crossing round (an unmeasured multi-patch decoder/pooling cost).
P0 is never changed. Gain = 1 - ms(P4) / ms(P0), both with the K chosen on the extended set.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import aligned_costmodel as cm  # noqa: E402

OUT = ROOT / "results/aligned"


def pred(t, I, extra_cross_ms: float, slope_mult: float) -> float:
    loc = 0.0
    rounds_cross = 0
    for key, n in t["by_kp"].items():
        kk, p, crossing, _ = (int(x) for x in key.split(","))
        b = I.B[p] if p <= 1 else I.B[1] + slope_mult * (I.B[p] - I.B[1])
        loc += n * (I.pair[kk] + b + (I.dp if p else 0.0))
        if crossing:
            rounds_cross += n
    host = t["lookup_raw_calls"] * I.host_comb + t["lookup_flagged_calls"] * (I.host_comb + cm.RETOK_US)
    total = loc + t["rounds"] * I.o_round + host / 1000 + rounds_cross * extra_cross_ms
    return I.prefill_pb + total / t["emitted"]


def main() -> int:
    md = ["# P4 sensitivity to the cost of a crossing round", "",
          "| precision | variant | P4 K | ms/byte ext | vs P0 ext | ms/byte default | vs P0 default |", "|---|---|---:|---:|---:|---:|---:|"]
    for prec in ("bf16", "int4"):
        I = cm.Inputs(prec)
        T = json.loads((OUT / f"tallies_{prec}_lookup.json").read_text(encoding="utf-8"))["cells"]
        p0 = {s: min(pred(T[f"{s}|-|cut|{k}|P0"], I, 0, 1) for k in cm.KS_EVAL) for s in ("ext", "default")}
        for label, extra, mult in (("as measured", 0.0, 1.0), ("B(P) slope x2", 0.0, 2.0), ("B(P) slope x3", 0.0, 3.0),
                                   ("+0.5 ms per crossing round", 0.5, 1.0), ("+1 ms per crossing round", 1.0, 1.0),
                                   ("+2 ms per crossing round", 2.0, 1.0)):
            k = min(cm.KS_EVAL, key=lambda kk: pred(T[f"ext|-|cut|{kk}|P4"], I, extra, mult))
            e, d = pred(T[f"ext|-|cut|{k}|P4"], I, extra, mult), pred(T[f"default|-|cut|{k}|P4"], I, extra, mult)
            md.append(f"| {prec} | {label} | {k} | {e:.3f} | {(1 - e / p0['ext']) * 100:+.1f}% | {d:.3f} | {(1 - d / p0['default']) * 100:+.1f}% |")
    (OUT / "sensitivity.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
