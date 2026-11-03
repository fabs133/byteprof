#!/bin/bash
# Drafter diagnostics: record drafter continuations at every byte of OLMo's output (3 jobs, ~2-3 min each).
cd "$(dirname "$0")/../.." || exit 1
for n in smollm2-360m qwen2.5-0.5b smollm2-135m; do
  .venv/Scripts/python.exe -X utf8 tools/diag_record_olmo_bytes.py --name $n > results/diagnostics/logs/rec-$n.log 2>&1
done
echo ALLDONE > results/diagnostics/logs/rec.done
