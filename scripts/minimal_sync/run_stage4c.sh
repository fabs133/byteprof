#!/bin/bash
# Rerun of the OLMo cells of Stage 4 (the first hf pair differed by 7.5%, over the 5% limit).
cd "$(dirname "$0")/../.." || exit 1
source .venv/Scripts/activate
run() { echo "=== $*" ; python -m byteprof profile "$@" 2>&1 | grep -E "ERROR|results written|Traceback" ; }
for rep in 1 2 3; do
  run --config configs/olmo2-1b-compile.toml --output-dir results/minsync/s4c-olmo-hf
  run --config configs/olmo2-1b-compile-minimal.toml --output-dir results/minsync/s4c-olmo-minimal
done
echo DONE
