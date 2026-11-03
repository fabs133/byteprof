#!/bin/bash
# usage: run_skinny_quality_all.sh bpb | cute
cd "$(dirname "$0")/../.." || exit 1
if [ "$1" = bpb ]; then
  for v in bf16sk g8 g8l4 all4; do bash scripts/skinny/run_skinny_quality.sh configs/skinny/bolmo-$v-none.toml bpb "" bolmo-$v-bpb; done
  bash scripts/skinny/run_skinny_quality.sh configs/draft/olmo-bf16-none.toml bpb "" olmo-bf16-bpb
  bash scripts/skinny/run_skinny_quality.sh configs/skinny/olmo-int8sk-none.toml bpb "" olmo-int8sk-bpb
else
  bash scripts/skinny/run_skinny_quality.sh configs/p4/bf16-none.toml cute 1500 bolmo-bf16-cute
  bash scripts/skinny/run_skinny_quality.sh configs/skinny/bolmo-g8-none.toml cute 1500 bolmo-g8-cute
fi
echo DONE > results/skinny/logs/quality_all_$1.done
