#!/bin/bash
# H59 second part (TASK_skinny_kernels.md, Stage 2.4). usage: run_skinny_exact.sh <variant> <prec: fp32|bf16> <truth|exact> [lengths]
# truth: record the non-drafted minimal-loop truth of the variant; exact: run P4 against it (bytes and boundary flags).
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
v=$1; prec=$2; step=$3; lens=${4:-256,1024}
mkdir -p results/skinny/exact results/skinny/truth
tag=$prec-$v
if [ "$prec" = fp32 ]; then none=configs/skinny/fp32-$v-none.toml; p4=configs/skinny/fp32-$v-p4.toml; else none=configs/skinny/bolmo-$v-none.toml; p4=configs/skinny/bolmo-$v-p4.toml; fi
if [ "$step" = truth ]; then
  $PY -X utf8 tools/draft_truth.py --config $none --tag $tag --lengths $lens --patch-limit 440 > results/skinny/exact/truth-$tag.log 2>&1
  mv results/drafting/truth/$tag.json results/skinny/truth/$tag.json
else
  $PY -X utf8 tools/draft_exact.py --config $p4 --truth results/skinny/truth/$tag.json --lengths $lens --out results/skinny/exact/$tag.json > results/skinny/exact/$tag.log 2>&1
fi
echo DONE > results/skinny/exact/$tag-$step.done
