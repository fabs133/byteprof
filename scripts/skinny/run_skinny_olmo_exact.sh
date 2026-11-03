#!/bin/bash
# OLMo 2 with int8 skinny blocks: fp32 exactness of combined lookup K = 8 against the non-drafted minimal loop (same weights),
# then bf16 divergences. usage: run_skinny_olmo_exact.sh <fp32|bf16>
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
p=$1
mkdir -p results/skinny/exact results/skinny/truth
if [ "$p" = fp32 ]; then none=configs/skinny/fp32-olmo-int8sk-none.toml; k8=configs/skinny/fp32-olmo-int8sk-k8.toml; else none=configs/skinny/olmo-int8sk-none.toml; k8=configs/skinny/olmo-int8sk-k8.toml; fi
$PY -X utf8 tools/olmo_truth.py --config $none --tag olmo-$p-int8sk --lengths 72,288 > results/skinny/exact/truth-olmo-$p.log 2>&1
mv results/drafting/truth/olmo-$p-int8sk.json results/skinny/truth/olmo-$p-int8sk.json
$PY -X utf8 tools/olmo_exact.py --config $k8 --truth results/skinny/truth/olmo-$p-int8sk.json --out results/skinny/exact/olmo-$p-int8sk.json > results/skinny/exact/olmo-$p.log 2>&1
echo DONE > results/skinny/exact/olmo-$p.done
