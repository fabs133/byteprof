#!/bin/bash
# usage: run_skinny_batch.sh <batch> <run>   (cell lists per batch; run 2 reverses the order)
cd "$(dirname "$0")/../.." || exit 1
case $1 in
  B2) CELLS="skinny/bolmo-bf16sk-none skinny/bolmo-bf16sk-p4 skinny/bolmo-g8l4-none skinny/bolmo-g8l4-p4";;
  B3) CELLS="p4/int4-none p4/int4-p4 skinny/bolmo-all4-none skinny/bolmo-all4-p4";;
  B4) CELLS="skinny/bolmo-g8-none skinny/bolmo-g8-p4 skinny/olmo-int8sk-none skinny/olmo-int8sk-k8 draft/olmo-bf16-none draft/olmo-bf16-combined-k8";;
esac
[ "$2" = 2 ] && CELLS=$(echo $CELLS | tr ' ' '\n' | tac | tr '\n' ' ')
bash scripts/skinny/run_skinny_speed.sh $1 $2 $CELLS
