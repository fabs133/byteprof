#!/bin/bash
# TASK_fair_ratio_floor.md: speed runs (Stage A/B), then quality (Stage A optional, Stage B). One GPU job at a time.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
run() { echo "=== $1 $(date +%H:%M:%S)"; $PY -X utf8 -m byteprof profile --config "$2" --output-dir "results/fair/$1" 2>&1 | grep -E "ERROR|results written|Traceback|quantized" | cut -c1-200; }
ev() { echo "=== eval $1 $2 $(date +%H:%M:%S)"; $PY -X utf8 -m byteprof eval --config "$1" --task $2 ${3:+--limit $3} --output-dir "results/fair/eval-$4" 2>&1 | grep -E "ERROR|eval written|Traceback" | cut -c1-200; }
B="bf16 local-int8 local-int4 global-int4 all-int4 global-int8 all-int8 l4g8"
O="bf16 int4-blocks int4-blocks-head int8-blocks"
for v in $B; do run s-bolmo-$v-r1 configs/fair/bolmo-$v.toml; done
for v in $O; do run s-olmo-$v-r1 configs/fair/olmo-$v.toml; done
for v in $(echo $O | tr ' ' '\n' | tac); do run s-olmo-$v-r2 configs/fair/olmo-$v.toml; done
for v in $(echo $B | tr ' ' '\n' | tac); do run s-bolmo-$v-r2 configs/fair/bolmo-$v.toml; done
echo SPEED_DONE
for v in bf16 local-int4 global-int8 all-int8 l4g8; do
  ev configs/fair/bolmo-$v.toml bpb "" bolmo-$v-bpb
  ev configs/fair/bolmo-$v.toml cute 1500 bolmo-$v-cute
done
for v in $O; do ev configs/fair/olmo-$v.toml bpb "" olmo-$v-bpb; done
echo DONE
