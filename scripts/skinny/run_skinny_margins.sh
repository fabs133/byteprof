#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
for v in g8 all4; do
  $PY -X utf8 tools/draft_margin_check.py --config configs/skinny/bolmo-$v-none.toml --exact results/skinny/exact/bf16-$v.json --out results/skinny/exact/margins-bf16-$v.json > results/skinny/exact/margins-bf16-$v.log 2>&1
done
echo DONE > results/skinny/exact/margins.done
