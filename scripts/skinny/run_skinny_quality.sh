#!/bin/bash
# Quality part of TASK_skinny_kernels.md (Stage 2.3): usage run_skinny_quality.sh <config> <bpb|cute> <limit or ""> <out name>
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/Scripts/python.exe
mkdir -p results/skinny/logs
out=results/skinny/eval-$4
[ -f "$out/eval.json" ] && { echo "skip $4 (done)"; exit 0; }
echo "=== eval $1 $2 $(date +%H:%M:%S)" >> results/skinny/logs/quality.log
$PY -X utf8 -m byteprof eval --config "$1" --task $2 ${3:+--limit $3} --output-dir "$out" 2>&1 | grep -E "ERROR|eval written|Traceback|bpb|accuracy" | cut -c1-200 >> results/skinny/logs/quality.log
echo "=== done $4 $(date +%H:%M:%S)" >> results/skinny/logs/quality.log
echo DONE > results/skinny/logs/quality_$4.done
