#!/bin/bash
# bf16 skinny traces without NVTX (the importer fails on the NVTX variant of this trace): one K per trace
cd "$(dirname "$0")/../.." || exit 1
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2023.4.4/target-windows-x64/nsys.exe"
for K in 0 6; do
  "$NSYS" profile --trace=cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop -o results/megakernel/nsys/bf16sk-k$K --force-overwrite true \
    .venv/Scripts/python.exe -X utf8 tools/nsys_rounds.py --child --config configs/skinny/bolmo-bf16sk-p4.toml --ks $K --reps 30 > results/megakernel/nsys/bf16sk-k$K.log 2>&1
  "$NSYS" export --type=sqlite --force-overwrite=true -o results/megakernel/nsys/bf16sk-k$K.sqlite results/megakernel/nsys/bf16sk-k$K.nsys-rep >> results/megakernel/nsys/bf16sk-k$K.log 2>&1
done
touch results/megakernel/nsys/stage0c.done
