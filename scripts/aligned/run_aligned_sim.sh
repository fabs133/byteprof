#!/bin/bash
# CPU only. Policies P0-P4 / L1 on recorded truth and drafter continuations (TASK_patch_aligned.md).
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
$PY -X utf8 tools/aligned_sim.py --prec bf16 --names lookup
$PY -X utf8 tools/aligned_sim.py --prec int4 --names lookup
for n in smollm2-135m smollm2-360m qwen2.5-0.5b; do
  $PY -X utf8 tools/aligned_sim.py --prec bf16 --names $n
done
for n in smollm2-135m smollm2-360m qwen2.5-0.5b; do
  $PY -X utf8 tools/aligned_sim.py --prec int4 --names $n
done
echo DONE
