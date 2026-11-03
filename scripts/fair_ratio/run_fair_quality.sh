#!/bin/bash
# Quality part of TASK_fair_ratio_floor.md (Stage B; Stage A optional OLMo bpb). Run after the speed phase and Stage C.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
ev() { [ -f "results/fair/eval-$4/eval.json" ] && { echo "skip $4 (done)"; return; }; echo "=== eval $1 $2 $(date +%H:%M:%S)"; $PY -X utf8 -m byteprof eval --config "$1" --task $2 ${3:+--limit $3} --output-dir "results/fair/eval-$4" 2>&1 | grep -E "ERROR|eval written|Traceback" | cut -c1-200; }
for v in bf16 local-int4 global-int8 all-int8 l4g8; do
  ev configs/fair/bolmo-$v.toml bpb "" bolmo-$v-bpb
  ev configs/fair/bolmo-$v.toml cute 1500 bolmo-$v-cute
done
for v in bf16 int4-blocks int4-blocks-head int8-blocks; do ev configs/fair/olmo-$v.toml bpb "" olmo-$v-bpb; done
echo DONE
