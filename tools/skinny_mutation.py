"""Mutation check of the skinny kernel tests: inject plausible bugs into byteprof/optim/skinny.py, run tests/test_skinny.py and
tests/test_skinny_integration.py, and require at least one failure per mutant; the file is restored afterwards.
Usage: python tools/skinny_mutation.py  -> results/skinny/mutation.json"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "byteprof/optim/skinny.py"
MUTANTS = [
    ("int4 nibble halves swapped", "wlo = ((((packed & 0xF)", "wlo = (((((packed >> 4) & 0xF)"),
    ("int4 zero offset 127 instead of 128", "+ 128.0\n            wlo", "+ 127.0\n            wlo"),
    ("int4 group scale ignored", "acc += sc[:, None] * part", "acc += part"),
    ("int8 channel scale ignored", "acc = acc * sc[:, None]", "acc = acc * 1.0"),
    ("split-K reduce skips the last split", "for i in tl.static_range(SPLIT):", "for i in tl.static_range(SPLIT - 1):"),
    ("split-K ranges overlap", "(sid + 1) * NB // SPLIT, num_stages", "(sid + 1) * NB // SPLIT + 1, num_stages"),
    ("K tail not masked on weights and activations",
     "mask=nmask[:, None] & kmask[None, :], other=0,\n                        eviction_policy=\"evict_first\")\n            x = tl.load(x_ptr + rm[None, :] * K + rk[:, None], mask=mmask[None, :] & kmask[:, None], other=0.0)",
     "mask=nmask[:, None], other=0,\n                        eviction_policy=\"evict_first\")\n            x = tl.load(x_ptr + rm[None, :] * K + rk[:, None], mask=mmask[None, :], other=0.0)"),
    ("row bucket too small (17 rows in a 16-row block)", "return 16 if m <= 16 else 32 if m <= 32 else 64", "return 16 if m <= 17 else 32 if m <= 32 else 64"),
    ("store not masked on rows >= M (direct store)", "mask=nmask[:, None] & mmask[None, :])\n    else:  # fp32 partial", "mask=nmask[:, None])\n    else:  # fp32 partial"),
    ("output rows transposed", "tl.store(y_ptr + rm[None, :] * N + rn[:, None], acc.to", "tl.store(y_ptr + rm[None, :] * N + rn[:, None] + 1, acc.to"),
    ("int4 pack: high nibble from code j + 32", "g[:, :, :, 1] << 4", "g[:, :, :, 0] << 4"),
]


def run() -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "-q", "-x", "tests/test_skinny.py", "tests/test_skinny_integration.py"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode != 0, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]


def main() -> int:
    orig = open(F, encoding="utf-8", newline="").read()
    res = []
    try:
        for name, old, new in MUTANTS:
            if old not in orig:
                res.append({"mutant": name, "killed": None, "note": "pattern not found"})
                print(name, "PATTERN NOT FOUND", flush=True)
                continue
            open(F, "w", encoding="utf-8", newline="").write(orig.replace(old, new, 1))
            killed, last = run()
            res.append({"mutant": name, "killed": killed, "last": last})
            print(f"{name}: {'killed' if killed else 'SURVIVED'} ({last})", flush=True)
    finally:
        open(F, "w", encoding="utf-8", newline="").write(orig)
    (ROOT / "results/skinny/mutation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
