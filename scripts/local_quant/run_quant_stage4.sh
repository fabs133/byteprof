#!/bin/bash
# Stage 4 of TASK_local_quant.md: speed. Two runs per variant, round 2 in reverse order (so clock
# drift is not tied to one variant). Waits for Stage 3 to finish first (one GPU job at a time).
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
until grep -q DONE stage3_quant.log 2>/dev/null; do sleep 15; done
run() { # name config
  echo "=== $1 $(date +%H:%M:%S)"
  $PY -X utf8 -m byteprof profile --config "$2" --output-dir "results/quant/$1" 2>&1 | grep -E "ERROR|results written|Traceback|quantized"
}
BOLMO="bf16 local-int8 local-int4 global-int4 all-int4"
for v in $BOLMO; do run s4-bolmo-$v-r1 configs/quant/bolmo-$v.toml; done
for v in $(echo $BOLMO | tr ' ' '\n' | tac); do run s4-bolmo-$v-r2 configs/quant/bolmo-$v.toml; done
echo BOLMO_DONE
for v in bf16 all-int4; do run s4-olmo-$v-r1 configs/quant/olmo-$v.toml; done
for v in all-int4 bf16; do run s4-olmo-$v-r2 configs/quant/olmo-$v.toml; done
echo DONE
