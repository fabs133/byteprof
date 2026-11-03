#!/bin/bash
# Drafter diagnostics: offline analyses for the given modes (CPU only).
cd "$(dirname "$0")/../.." || exit 1
for m in "$@"; do
  .venv/Scripts/python.exe -X utf8 tools/diag_analyze.py --mode $m > results/diagnostics/logs/analyze-$m.log 2>&1
done
echo ALLDONE > results/diagnostics/logs/analyze-$1.done
