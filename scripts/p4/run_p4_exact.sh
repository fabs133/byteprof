#!/bin/bash
# H51 (TASK_build_p4.md): fp32 exactness of the gated loop against the recorded minimal-loop truth. usage: run_p4_exact.sh <256|1024>
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
n=$1
$PY -X utf8 tools/draft_exact.py --config configs/p4/fp32-p4.toml --truth results/drafting/truth/fp32.json --lengths $n \
  --out results/p4/exact/fp32-p4-$n.json > results/p4/exact/fp32-p4-$n.log 2>&1
echo DONE > results/p4/exact/fp32-p4-$n.done
