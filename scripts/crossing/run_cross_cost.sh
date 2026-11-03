#!/bin/bash
# Task B, B0.4: drafter round cost, one model after the other (each well under 10 minutes)
cd "$(dirname "$0")/../.." || exit 1
for n in smollm2-135m smollm2-360m qwen2.5-0.5b; do
  .venv/Scripts/python.exe -X utf8 tools/cross_drafter_cost.py --name $n --cs 1,3,6 --ms 1,2,4,8,12 --reps 40 > results/crossing/b0/logs/cost-$n.log 2>&1
done
echo ALLDONE > results/crossing/b0/logs/cost-all.done
