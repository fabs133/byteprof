#!/bin/bash
# Stage 0 of TASK_minimal_sync_loop: graph-global + hf loop, ms/byte vs length, with and without instrumentation.
cd "$(dirname "$0")/../.." || exit 1
source .venv/Scripts/activate
run() { echo "=== $*" ; python -m byteprof profile "$@" 2>&1 | grep -E "ERROR|results written|Traceback" ; }
for n in 64 256 1024; do
  for rep in 1 2; do
    run --config configs/bolmo-1b-graph-global.toml --max-new-bytes $n --output-dir results/minsync/s0-plain-$n
    run --config configs/bolmo-1b-graph-global.toml --max-new-bytes $n --instrument timing --output-dir results/minsync/s0-timing-$n
  done
done
run --config configs/bolmo-1b-graph-global.toml --max-new-bytes 256 --runs 1 --instrument syncs --output-dir results/minsync/s0-syncs-256
echo DONE
