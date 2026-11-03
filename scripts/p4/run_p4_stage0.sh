#!/bin/bash
# Stage 0 of TASK_build_p4.md: crossing-round microbenchmark, bf16 and all-int4 (K = 8 was run first on its own)
cd "$(dirname "$0")/../.." || exit 1
mkdir -p results/p4/stage0/logs
PY=.venv/Scripts/python.exe
for spec in "bf16 16" "bf16 32" "int4 8" "int4 16" "int4 32"; do
  set -- $spec
  $PY -X utf8 tools/p4_cross_bench.py --config configs/p4/$1-bench.toml --prec $1 --ks $2 --reps 120 > results/p4/stage0/logs/$1-k$2.log 2>&1
done
echo DONE > results/p4/stage0/logs/DONE
