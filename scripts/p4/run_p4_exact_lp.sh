#!/bin/bash
# bf16 / int4 divergences of the gated loop against the recorded minimal loop, then the top-2 logit gap at the first divergences.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
for prec in bf16 int4; do
  for n in 256 1024; do
    $PY -X utf8 tools/draft_exact.py --config configs/p4/$prec-p4.toml --truth results/drafting/truth/$prec.json --lengths $n \
      --out results/p4/exact/$prec-p4-$n.json > results/p4/exact/$prec-p4-$n.log 2>&1
  done
done
echo DONE > results/p4/exact/lp.done
