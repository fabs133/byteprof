#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2023.4.4/target-windows-x64/nsys.exe"
.venv/Scripts/python.exe -X utf8 tools/nsys_rounds.py --config configs/skinny/bolmo-bf16sk-p4.toml --out results/megakernel/nsys/bf16sk --ks 0,6 --reps 30 > results/megakernel/nsys/bf16sk.log 2>&1
"$NSYS" export --type=sqlite --force-overwrite=true -o results/megakernel/nsys/bf16sk.sqlite results/megakernel/nsys/bf16sk.nsys-rep >> results/megakernel/nsys/bf16sk.log 2>&1
touch results/megakernel/nsys/stage0b.done
