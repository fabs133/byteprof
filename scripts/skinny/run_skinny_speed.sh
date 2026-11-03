#!/bin/bash
# Stage 2 speed cells (TASK_skinny_kernels.md): usage run_skinny_speed.sh <batch> <run 1|2> <cell> [<cell> ...]
# Cells are config paths relative to configs/ without .toml (e.g. p4/bf16-p4, skinny/bolmo-g8-none). Run 2 is given the
# cells in reverse order by the caller. Results: results/skinny/speed/<batch>/<cell>-run<r>.json
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
batch=$1; run=$2; shift 2
mkdir -p results/skinny/speed/$batch results/skinny/logs
for c in "$@"; do
  n=$(basename $c)
  $PY -X utf8 tools/draft_speed.py --config configs/$c.toml --out results/skinny/speed/$batch/$n-run$run.json >> results/skinny/logs/speed_$batch.log 2>&1
done
echo DONE > results/skinny/logs/speed_${batch}_run${run}.done
