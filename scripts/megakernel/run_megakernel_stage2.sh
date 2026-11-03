#!/bin/bash
# Stage 2 of TASK_megakernel_theory.md: bf16, int8, int4 (each in its own process, < 3 min)
cd "$(dirname "$0")/../.." || exit 1
for q in bf16 int8 int4; do
  .venv/Scripts/python.exe -X utf8 tools/mk_stream_bench.py --q $q --rounds 7 > results/megakernel/stream-$q.log 2>&1
done
touch results/megakernel/stage2.done
