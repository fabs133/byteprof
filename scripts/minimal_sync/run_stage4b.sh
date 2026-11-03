#!/bin/bash
# Stage 4 of TASK_minimal_sync_loop. (The Bolmo hf-loop cells are the Stage 0 "plain" runs in results/minsync/s0-plain-*.)
cd "$(dirname "$0")/../.." || exit 1
source .venv/Scripts/activate
run() { echo "=== $*" ; python -m byteprof profile "$@" 2>&1 | grep -E "ERROR|results written|Traceback" ; }
for rep in 1 2; do
  for n in 64 256 1024; do
    run --config configs/bolmo-1b-minimal.toml --max-new-bytes $n --output-dir results/minsync/s4-bolmo-minimal-$n
  done
  run --config configs/olmo2-1b-compile.toml --output-dir results/minsync/s4-olmo-hf
  run --config configs/olmo2-1b-compile-minimal.toml --output-dir results/minsync/s4-olmo-minimal
done
for n in 256 1024; do
  run --config configs/bolmo-1b-minimal.toml --max-new-bytes $n --instrument timing --runs 2 --output-dir results/minsync/s4-bolmo-minimal-timing-$n
done
run --config configs/bolmo-1b-minimal.toml --runs 1 --trace --output-dir results/minsync/s4-bolmo-minimal-trace
run --config configs/bolmo-1b-graph-global.toml --runs 1 --trace --output-dir results/minsync/s4-bolmo-hf-trace
run --config configs/olmo2-1b-compile.toml --runs 1 --trace --output-dir results/minsync/s4-olmo-hf-trace
run --config configs/olmo2-1b-compile-minimal.toml --runs 1 --trace --output-dir results/minsync/s4-olmo-minimal-trace
echo DONE
