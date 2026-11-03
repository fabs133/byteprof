#!/bin/bash
# Stage 4 measurement sequence (kept for reproducibility). Run from the project root.
cd "$(dirname "$0")/../.." || exit 1
source .venv/Scripts/activate
run() { echo "=== $*" ; python -m byteprof profile "$@" 2>&1 | grep -E "ERROR|results written|Traceback" ; }
for rep in 1 2; do
  run --config configs/olmo2-1b.toml
  run --config configs/olmo2-1b-compile.toml
  run --config configs/bolmo-1b.toml
  run --config configs/bolmo-1b-graph.toml
  run --config configs/bolmo-1b-graph-global.toml
  run --config configs/bolmo-1b-compile.toml
done
run --config configs/olmo2-1b.toml --runs 1 --trace
run --config configs/olmo2-1b-compile.toml --runs 1 --trace
run --config configs/bolmo-1b.toml --runs 1 --trace
run --config configs/bolmo-1b-graph.toml --runs 1 --trace
run --config configs/bolmo-1b-graph-global.toml --runs 1 --trace
run --config configs/bolmo-1b-compile.toml --runs 1 --trace
echo DONE
