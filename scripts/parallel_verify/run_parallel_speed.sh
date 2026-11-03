#!/bin/bash
# Stage A2 end-to-end cells (TASK_parallel_verify.md): usage: run_parallel_speed.sh <bf16|int4> <1|2>
# Run 1 in the listed order, run 2 in the opposite order (each batch stays under 10 minutes).
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
prec=$1; run=$2
if [ "$prec" = bf16 ]; then CELLS="none steps-k6 fused-k6 fused-k8"; else CELLS="none steps-k4 fused-k4 fused-k6"; fi
[ "$run" = 2 ] && CELLS=$(echo $CELLS | tr ' ' '\n' | tac | tr '\n' ' ')
for c in $CELLS; do
  if [ "$c" = none ]; then cfg=configs/draft/bolmo-$prec-none.toml; else cfg=configs/parallel/speed-$prec-$c.toml; fi
  $PY -X utf8 tools/draft_speed.py --config $cfg --out results/parallel/speed/$prec-$c-run$run.json >> results/parallel/logs/speed_$prec.log 2>&1
done
echo DONE > results/parallel/logs/speed_${prec}_$run.done
