#!/bin/bash
# Stage 3 of TASK_local_quant.md: quality for bf16, local int8, local int4, global int4, all int4.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
for v in bf16 local-int8 local-int4 global-int4 all-int4; do
  for task in bpb cute; do
    echo "=== $v $task $(date +%H:%M:%S)"
    $PY -X utf8 -m byteprof eval --config configs/quant/bolmo-$v.toml --task $task \
        --output-dir results/quant/eval-$v-$task 2>&1 | grep -v -i "warn\|xet\|Loading\|torch_dtype" | tail -4
  done
done
echo DONE
