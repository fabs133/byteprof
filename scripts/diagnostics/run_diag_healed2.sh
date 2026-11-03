#!/bin/bash
# Drafter diagnostics: healed2 recordings (3 jobs, each well under 10 minutes).
cd "$(dirname "$0")/../.." || exit 1
for n in smollm2-360m qwen2.5-0.5b smollm2-135m; do
  .venv/Scripts/python.exe -X utf8 tools/diag_record_healed2.py --name $n > results/diagnostics/logs/h2-$n.log 2>&1
done
echo ALLDONE > results/diagnostics/logs/h2.done
