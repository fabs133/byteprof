#!/bin/bash
# Task B: record the LM continuations on the all-int4 ground truth (the int4 text differs from the bf16 text)
cd "$(dirname "$0")/../.." || exit 1
for n in smollm2-135m smollm2-360m qwen2.5-0.5b; do
  .venv/Scripts/python.exe -X utf8 tools/cross_record_drafts.py --name $n --truth-tag int4 --no-olmo > results/crossing/b0/logs/rec-int4-$n.log 2>&1
done
echo ALLDONE > results/crossing/b0/logs/rec-int4.done
