"""H59 (first part): worst componentwise error max_ij |y - ref|_ij / (|x| |W|^T)_ij of the skinny kernels against the fp32
dequantize-then-matmul reference, for every M = 1..33 and every real layer shape (random weights, std 0.02).
Output: results/skinny/errors.json"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import torch  # noqa: E402

from byteprof.optim import skinny as S  # noqa: E402
from test_skinny import REAL, make, nerr  # noqa: E402

out = {}
for kind in ("bf16", "int8", "int4"):
    worst = {}
    for n, k in REAL:
        _, args = make(kind, n, k)
        ref_w = S.dequantize(kind, *args)
        w = 0.0
        for m in range(1, 34):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            w = max(w, nerr(S.skinny_mm(x, *args), x, ref_w))
        worst[f"{n}x{k}"] = w
    out[kind] = {"worst": max(worst.values()), "by_shape": worst}
    print(kind, f"{out[kind]['worst']:.2e}", flush=True)
Path(ROOT / "results/skinny/errors.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
