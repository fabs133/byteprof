#!/bin/bash
# Extended prompt set, one pass: acceptance of the gated loop (bf16 and int4).
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
mkdir -p results/p4/speed/ext results/p4/logs
for prec in bf16 int4; do
  $PY -X utf8 tools/draft_speed.py --config configs/p4/$prec-p4-ext.toml --out results/p4/speed/ext/$prec-p4-ext.json >> results/p4/logs/speed_ext.log 2>&1
done
echo DONE > results/p4/logs/ext.done
