#!/bin/bash
# Stage 1 tuning of the skinny kernels (one kind per call, each under ~10 minutes)
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
for kind in "$@"; do
  $PY -X utf8 tools/skinny_tune.py --kinds $kind > results/skinny/tune_$kind.log 2>&1
done
echo DONE > results/skinny/tune.done
