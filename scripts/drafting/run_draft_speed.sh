#!/bin/bash
# Stage 3 speed table for Bolmo (TASK_drafting.md): two runs per cell, the second in the opposite order.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
for prec in bf16 int4; do
  if [ "$prec" = bf16 ]; then CELLS="none combined-k6 oracle-k6 combined-k4 combined-k8 context-k4"; else CELLS="none combined-k4 oracle-k4 combined-k2 combined-k6 context-k4"; fi
  REV=$(echo $CELLS | tr ' ' '\n' | tac | tr '\n' ' ')
  for c in $CELLS; do $PY -X utf8 tools/draft_speed.py --config configs/draft/bolmo-$prec-$c.toml --out results/drafting/speed/$prec/$c-run1.json >> results/drafting/logs/speed_$prec.log 2>&1; done
  for c in $REV; do $PY -X utf8 tools/draft_speed.py --config configs/draft/bolmo-$prec-$c.toml --out results/drafting/speed/$prec/$c-run2.json >> results/drafting/logs/speed_$prec.log 2>&1; done
done
echo DONE > results/drafting/logs/speed_all.done
