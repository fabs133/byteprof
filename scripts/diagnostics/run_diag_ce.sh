#!/bin/bash
# Drafter diagnostics: teacher-forced cross-entropy (3 short GPU jobs).
cd "$(dirname "$0")/../.." || exit 1
for n in smollm2-360m qwen2.5-0.5b smollm2-135m; do
  .venv/Scripts/python.exe -X utf8 tools/diag_ce.py --name $n > results/diagnostics/logs/ce-$n.log 2>&1
done
echo ALLDONE > results/diagnostics/logs/ce.done
