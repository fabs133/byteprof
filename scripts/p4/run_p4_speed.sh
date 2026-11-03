#!/bin/bash
# Stage 2 speed cells (TASK_build_p4.md): usage: run_p4_speed.sh <bf16|int4> <1|2> <cell> [<cell> ...]
# Cells are configs/p4/<prec>-<cell>.toml. Run 2 is given the cells in reverse order by the caller.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
prec=$1; run=$2; shift 2
mkdir -p results/p4/speed results/p4/logs
for c in "$@"; do
  $PY -X utf8 tools/draft_speed.py --config configs/p4/$prec-$c.toml --out results/p4/speed/$prec-$c-run$run.json >> results/p4/logs/speed_$prec.log 2>&1
done
echo DONE > results/p4/logs/speed_${prec}_run${run}.done
