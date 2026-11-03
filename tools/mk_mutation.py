"""Mutation check of tests/test_megakernel_bench.py: inject plausible bugs into the CUDA sources and confirm a test fails.

  python -X utf8 tools/mk_mutation.py [--out results/megakernel/mutation.json]
Each mutant patches one string in byteprof/bench/{stream,sync}_src.py, runs the test file (-x -q), and restores the file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STREAM = ROOT / "byteprof/bench/stream_src.py"
SYNC = ROOT / "byteprof/bench/sync_src.py"

MUTANTS = [
    ("barrier base counts nph instead of nph - 1 (second launch times out)", STREAM,
     "(unsigned)(nph - 1) * (unsigned)G;", "(unsigned)nph * (unsigned)G;"),
    ("int4: subtract the 8 * sum(x) offset per group instead of per vector", STREAM,
     "- 8.f * s.xsum[j]);", "- 8.f * s.xsum[j >> 2]);"),
    ("n / m double buffer: always read buffer 0", STREAM,
     "const float* nsrc = par ? ph.n2 : ph.n;", "const float* nsrc = ph.n;"),
    ("prefetch: staged row copied one 16-byte chunk off", STREAM,
     "s.sw + (size_t)row * ph.rowstride + 16 * c);", "s.sw + (size_t)row * ph.rowstride + 16 * c + 16);"),
    ("persistent kernel does not wait at the barrier", STREAM,
     "if (t == 0) { wait_ge<false>(counter, base + (unsigned)(p + 1) * G, err, cap_ns); stop = (int)*err; }",
     "if (t == 0) { stop = 0; }"),
    ("residual add dropped in the epilogue", STREAM, "__ldcg(ph.res + rr[q]) + v;", "v;"),
    ("mLSTM forget gate dropped from the n update", STREAM, "(fp * nn.x + ip * 0.125f * kk.x)", "(nn.x + ip * 0.125f * kk.x)"),
    ("silu dropped from the gate / up pair", STREAM, "(g / (1.f + __expf(-g))) * u;", "g * u;"),
    ("int8 per-row scale dropped", STREAM, "if (Q == 1) v *= ph.m[mi[q]].rs[rr[q]];", ""),
    ("rows of the last block dropped (partition end off by one)", STREAM,
     "const int v1 = block_start(ph.rows_total / unit, b + 1, G) * unit;\n  const int K = ph.K;",
     "const int v1 = block_start(ph.rows_total / unit, b + 1, G) * unit - (b == G - 1 ? 1 : 0);\n  const int K = ph.K;"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/megakernel/mutation.json")
    a = ap.parse_args()
    res = []
    for name, path, old, new in MUTANTS:
        src = path.read_text(encoding="utf-8")
        assert src.count(old) == 1, (name, src.count(old))
        path.write_text(src.replace(old, new), encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "tests/test_megakernel_bench.py", "-x", "-q"],
                               cwd=ROOT, capture_output=True, text=True, timeout=170)
            killed = r.returncode != 0
            tail = (r.stdout.strip().splitlines() or [""])[-1]
        except subprocess.TimeoutExpired:
            killed, tail = True, "timeout"
        finally:
            path.write_text(src, encoding="utf-8")
        res.append({"mutant": name, "killed": killed, "last_line": tail})
        print(("KILLED  " if killed else "SURVIVED"), name, "|", tail, flush=True)
    Path(ROOT / a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{sum(r['killed'] for r in res)}/{len(res)} killed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
