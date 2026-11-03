#!/bin/bash
# Piece benchmarks (verify pairs and global passes) of a skinny variant: usage run_skinny_pieces.sh <variant>...
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
mkdir -p results/skinny/bench results/skinny/logs
for v in "$@"; do
  $PY -X utf8 tools/draft_verify_bench.py --config configs/skinny/bench-$v.toml --tag $v --accept none --root results/skinny/bench >> results/skinny/logs/pieces_$v.log 2>&1
  $PY -X utf8 tools/draft_global_multi_bench.py --config configs/skinny/bench-$v.toml --tag $v --max-p 8 >> results/skinny/logs/pieces_$v.log 2>&1
done
echo DONE > results/skinny/logs/pieces.done
