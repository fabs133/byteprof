#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
$PY -X utf8 tools/draft_margin_check.py --config configs/draft/bolmo-bf16-none.toml --exact results/p4/exact/bf16-p4-1024.json --out results/p4/exact/margins-bf16.json > results/p4/exact/margins-bf16.log 2>&1
$PY -X utf8 tools/draft_margin_check.py --config configs/draft/bolmo-int4-none.toml --exact results/p4/exact/int4-p4-1024.json --out results/p4/exact/margins-int4.json > results/p4/exact/margins-int4.log 2>&1
echo DONE > results/p4/exact/margins.done
