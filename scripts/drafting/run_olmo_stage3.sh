#!/bin/bash
# Stage 3, OLMo 2 side (TASK_drafting.md): fp32 exactness, speed cells (two runs, second in reverse order), then the H27 block.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
L=results/drafting/logs
$PY -X utf8 tools/olmo_truth.py --config configs/draft/olmo-fp32-none.toml --tag olmo-fp32 --lengths 72,288 > $L/olmo_truth_fp32.log 2>&1
for k in combined-k8 context-k8 combined-k4; do
  $PY -X utf8 tools/olmo_exact.py --config configs/draft/olmo-fp32-$k.toml --truth results/drafting/truth/olmo-fp32.json --out results/drafting/exact/olmo-fp32-$k.json > $L/olmo_exact_fp32_$k.log 2>&1
done
CELLS="none combined-k8 combined-k4 combined-k2 context-k8"
REV=$(echo $CELLS | tr ' ' '\n' | tac | tr '\n' ' ')
for c in $CELLS; do $PY -X utf8 tools/draft_speed.py --config configs/draft/olmo-bf16-$c.toml --out results/drafting/speed/olmo-bf16/$c-run1.json >> $L/speed_olmo.log 2>&1; done
for c in $REV; do $PY -X utf8 tools/draft_speed.py --config configs/draft/olmo-bf16-$c.toml --out results/drafting/speed/olmo-bf16/$c-run2.json >> $L/speed_olmo.log 2>&1; done
# H27: both models, drafted and not, interleaved in one block (run 1 in this order, run 2 reversed)
ORDER="bolmo-bf16-none bolmo-bf16-combined-k6 olmo-bf16-none olmo-bf16-combined-k8"
RORDER="olmo-bf16-combined-k8 olmo-bf16-none bolmo-bf16-combined-k6 bolmo-bf16-none"
for c in $ORDER; do $PY -X utf8 tools/draft_speed.py --config configs/draft/$c.toml --out results/drafting/speed/h27/$c-run1.json >> $L/speed_h27.log 2>&1; done
for c in $RORDER; do $PY -X utf8 tools/draft_speed.py --config configs/draft/$c.toml --out results/drafting/speed/h27/$c-run2.json >> $L/speed_h27.log 2>&1; done
echo DONE > $L/olmo_all.done
