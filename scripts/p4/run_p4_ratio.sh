#!/bin/bash
# H53 block (TASK_build_p4.md): Bolmo minimal and P4, OLMo 2 minimal and combined lookup K = 8, bf16, one session, interleaved.
# Run 1 in this order, run 2 reversed. Each run is under 10 minutes.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
mkdir -p results/p4/speed/h53 results/p4/logs
run=$1
ORDER="draft/bolmo-bf16-none p4/bf16-p4 draft/olmo-bf16-none draft/olmo-bf16-combined-k8"
[ "$run" = 2 ] && ORDER=$(echo $ORDER | tr ' ' '\n' | tac | tr '\n' ' ')
for c in $ORDER; do
  n=$(basename $c)
  $PY -X utf8 tools/draft_speed.py --config configs/$c.toml --out results/p4/speed/h53/$n-run$run.json >> results/p4/logs/speed_h53.log 2>&1
done
echo DONE > results/p4/logs/h53_run$run.done
