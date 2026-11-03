# Study scripts

One folder per study; each study has a section in `FINDINGS.md`. Every script changes to the repository
root itself (`cd "$(dirname "$0")/../.." || exit 1`), so it can be started from any directory. The scripts
use the Windows venv layout (`.venv/Scripts/python.exe` or `.venv/Scripts/activate`) and write to `results/`
(not in git). Run them with `bash scripts/<study>/<script>.sh`, one GPU job at a time and with Ollama stopped.
The task files that defined the studies are not shipped.

| Folder | What it ran | `FINDINGS.md` section |
|---|---|---|
| `cuda_graphs` | Eager vs CUDA-graph vs compiled variants of Bolmo and OLMo 2 (twice, plus one trace each) | Overhead removal |
| `minimal_sync` | Stage 0 / Stage 4 measurement sequences of the one-sync decode loop | Minimal-sync loop |
| `local_quant` | Quality (bpb, CUTE) and speed of int8/int4 quantization variants | Local-stack quantization |
| `fair_ratio` | Matched-kernel Bolmo/OLMo speed cells and quality evals | Fair ratio, global int8, floor |
| `drafting` | Within-patch drafting speed cells (Bolmo) and the OLMo 2 side (fp32 exactness, speed) | Drafting, Drafting: report |
| `parallel_verify` | End-to-end cells of the fused multi-step verification | Parallel local verification |
| `crossing` | Small-LM drafter round cost and int4 recordings | Crossing drafts |
| `diagnostics` | Drafter recordings, cross-entropy and offline analyses | Drafter diagnostics |
| `aligned` | CPU simulation of patch-aligned policies | Patch-aligned rounds |
| `p4` | Boundary-gated drafting (P4): microbenchmark, speed, exactness, margins, ratio | Boundary-gated drafting (P4) |
| `skinny` | Skinny Triton matmuls: tuning, piece benchmarks, speed, exactness, quality | Skinny kernels |
| `megakernel` | Nsight traces and stream benchmarks for the megakernel question | Megakernel theory |

## Usage lines

Scripts not listed take no arguments.

- `p4/run_p4_exact.sh <256|1024>`
- `p4/run_p4_ratio.sh <run>` (run 1 in the listed order, run 2 reversed)
- `p4/run_p4_speed.sh <bf16|int4> <1|2> <cell> [<cell> ...]`
- `parallel_verify/run_parallel_speed.sh <bf16|int4> <1|2>`
- `diagnostics/run_diag_analyze.sh <mode> [<mode> ...]`
- `skinny/run_skinny_batch.sh <batch> <run>`
- `skinny/run_skinny_exact.sh <variant> <fp32|bf16> <truth|exact> [lengths]`
- `skinny/run_skinny_olmo_exact.sh <fp32|bf16>`
- `skinny/run_skinny_pieces.sh <variant> [<variant> ...]`
- `skinny/run_skinny_quality.sh <config> <bpb|cute> <limit or ""> <out name>`
- `skinny/run_skinny_quality_all.sh bpb | cute`
- `skinny/run_skinny_speed.sh <batch> <run 1|2> <cell> [<cell> ...]`
- `skinny/run_skinny_tune.sh <kind> [<kind> ...]`

`local_quant/run_quant_stage4.sh` waits until a file `stage3_quant.log` in the repository root contains `DONE`,
so start Stage 3 with its output redirected there: `bash scripts/local_quant/run_quant_stage3.sh > stage3_quant.log`.
