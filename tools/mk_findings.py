"""Appends the results part of `## Megakernel theory` to FINDINGS.md (the pre-registered hypotheses are already there).

Reads results/megakernel/*.json; verifies that the existing file is a prefix of the new one. CRLF line endings as in the file.
   python -X utf8 tools/mk_findings.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results/megakernel"
F = ROOT / "FINDINGS.md"
SNAP = ROOT / "results/FINDINGS.after_hypotheses_megakernel.md"


def J(name: str):
    return json.loads((R / name).read_text(encoding="utf-8"))


def main() -> int:
    sync, asym, ts = J("sync.json"), J("asym.json"), J("tsprof-bf16.json")
    st = {q: J(f"stream-{q}.json") for q in ("bf16", "int8", "int4")}
    cm = J("costmodel.json")
    c0, c6, c4 = J("census-bf16sk-k0.json"), J("census-bf16sk-k6.json"), J("census-int4.json")
    mut = J("mutation.json")
    rows = sync["rows"]
    sm, pred, fr = cm["step_models"], cm["predictions"], cm["floor_ratio"]
    nl, i4 = pred["near-lossless (global int8, local bf16 skinny)"], pred["all-int4 (tinygemm)"]
    b0, b6 = c0["blocks"]["0"], c6["blocks"]["6"]
    d0, d4 = c4["blocks"]["0"], c4["blocks"]["4"]

    def stream_row(q, name):
        v = st[q]["variants"][name]
        return v

    def best(q, prefix):
        items = [(k, v) for k, v in st[q]["variants"].items() if k.startswith(prefix)]
        return min(items, key=lambda kv: kv[1]["us_median"])

    bar, bar_full = rows["barrier, sync only (no read-back)"]["us_per_phase_slope"], rows["barrier (one counter)"]["us_per_phase_slope"]
    df1, df2, df4 = (rows[f"dataflow {n} producer{'s' if n > 1 else ''}, sync only"]["us_per_phase_slope"] for n in (1, 2, 4))
    df_full = {n: rows[f"dataflow {n} producer{'s' if n > 1 else ''}"]["us_per_phase_slope"] for n in (1, 2, 4)}
    gs, gn = rows["grid.sync()"]["us_per_phase_slope"], rows["graph of tiny kernels"]["us_per_phase_slope"]

    pre = {k: v for k, v in st["bf16"]["variants"].items() if k.startswith("c")}
    frac = lambda v: 100 * v["weight_GBs"] / 411.0   # noqa: E731
    bf_b = st["bf16"]["variants"]["b: persistent, no prefetch"]
    bf_c_best = max((v for k, v in pre.items() if k.startswith("c:") or k.startswith("c2")), key=lambda v: v["weight_GBs"])
    pp_best = frac(bf_c_best) - frac(bf_b)
    pp_worst = min(frac(v) for k, v in pre.items() if k.startswith(("c:", "c2"))) - frac(bf_b)

    def fr_cell(loop, prec):
        return next(r for r in fr["rows"] if r["loop"] == loop and r["precision"] == prec)

    dr_bf, dr_i8 = fr_cell("drafted bf16", "bf16"), fr_cell("drafted int8", "int8")
    mi_bf, mi_i8, mi_i4 = fr_cell("minimal", "bf16"), fr_cell("minimal", "int8"), fr_cell("minimal", "int4")
    g_nl, g_i4 = cm["gate"]["near-lossless (global int8, local bf16 skinny)"], cm["gate"]["all-int4 (tinygemm)"]
    nl_m, nl_i = nl["measured kernel"]["P4"], nl["ideal overlap"]["P4"]
    i4_m, i4_e, i4_i = i4["measured kernel"]["P4"], i4["extrapolated by bytes"]["P4"], i4["ideal overlap"]["P4"]
    nl_min, i4_min = nl["measured kernel"]["minimal"], i4["extrapolated by bytes"]["minimal"]
    mm_above = b0["matmul_us"] - c0["floor_us"]
    need_delta = 0.10 * nl_m["today_ms_per_byte"] / nl_m["rounds_per_byte"] * 1000

    L: list[str] = []
    A = L.append
    A("")
    A("### Environment and method")
    A("")
    A("- RTX 3060 Ti (38 SMs; torch reports a 3 MiB L2), driver 621.01, torch 2.6.0+cu124, CUDA toolkit 12.4 headers, one GPU visible. Ollama was not running; the GPU was idle (1-3% utilization, checked with nvidia-smi before the main runs). No hardware, driver or system setting was changed; the display watchdog (TDR) was left alone.")
    A("- Benchmark kernels are CUDA C++ compiled at run time with NVRTC (`nvrtc64_120_0.dll` of the installed 12.4 toolkit, straight to a cubin for sm_86) and launched through the driver API from ctypes (`byteprof/bench/cuda_rt.py`); no MSVC is needed (`cl` is not installed), and no new package was installed. The kernels use torch's context and stream, so they can be captured in torch CUDA graphs. Code: `byteprof/bench/{cuda_rt,sync_src,sync_bench,stream_src,stream_bench}.py`, drivers `tools/mk_*.py`, `run_megakernel_stage*.sh`, results `results/megakernel/`. No decode loop and no earlier section of this file was changed (`results/FINDINGS.before_megakernel.md`; the prefix was verified).")
    A("- Safety: persistent grids are checked against the occupancy API (`cuOccupancyMaxActiveBlocksPerMultiprocessor` x SM count) before each launch; every spin loop has a wall-clock cap (globaltimer; 50 ms in Stage 1, 20 ms in Stage 2) that sets an error flag, after which all blocks leave. The cap was exercised on purpose in tests and once by accident (a counter-base bug in my first version of the kernel timed out on the second launch and ended in the error flag, without a hang).")
    A("- The stack modelled in Stage 2 has the real shapes and parameter count (180.2 M parameters, 360.35 MB bf16): one encoder layer, `out_projection`, `in_projection`, four decoder layers, final norm and `lm_head`; five dependent phases per layer (norm + q/k/v/ogate/igate/fgate projections; mLSTM state update with the real 16 x 64 x 128 fp32 state; out projection + residual; norm + gate/up with silu; down + residual) = 28 phases and 27 barriers per step. Weights are random. The norm and mLSTM math is simplified (no per-head norm, no convolution; stabilised gates and the matrix-memory update are kept), the same device code runs in every variant, and a torch fp32 reference of the same math is the correctness check. Only M = 1 (K = 0) is measured.")
    A("")
    A("### 1. Verdicts")
    A("")
    A("| Hypothesis | Verdict | Numbers |")
    A("|---|---|---|")
    A(f"| **H60** census: bf16 K = 0 step >= 35% above the floor; >= 70% of the excess inside kernels | **Supported (narrowly on the first part)** | traced round {b0['period_us']:.0f} us vs floor {c0['floor_us']:.0f} us (360.35 MB at 411 GB/s): excess {b0['excess_us']:.0f} us = **{100 * b0['share_above_floor']:.0f}%** of the round; inside kernels {b0['inside_excess_us']:.0f} us = **{100 * b0['share_of_excess_inside']:.0f}%** of the excess, gaps {b0['between_us']:.0f} us. But the inside part is not ramp/tail of the matmuls (matmul kernels {b0['matmul_us']:.0f} us, only {mm_above:.0f} us over the floor): it is {b0['other_kernels']} small non-matmul kernels ({b0['other_us']:.0f} us). |")
    A(f"| **H61** sync: <= 2 us per phase; dataflow (<= 4 producers) >= 30% cheaper than an all-blocks barrier | **First part supported, second refuted** | barrier **{bar:.2f} us/phase** (sync only), {bar_full:.2f} us with the publication read-back; dataflow {df1:.2f} / {df2:.2f} / {df4:.2f} us for 1 / 2 / 4 producers = **{100 * (1 - df1 / bar):.0f}% / {100 * (1 - df2 / bar):.0f}% / {100 * (1 - df4 / bar):.0f}% cheaper** (with read-back: {df_full[1]:.2f} / {df_full[2]:.2f} / {df_full[4]:.2f} vs {bar_full:.2f}). |")
    A(f"| **H62** streaming: persistent bf16 chain with prefetch >= 75% of copy bandwidth, and prefetch adds >= 10 points | **First part supported for bf16 only, second refuted** | best persistent bf16 {sm['bf16']['measured_persistent_us']:.0f} us = {sm['bf16']['measured_GBs']:.0f} GB/s = **{100 * sm['bf16']['measured_GBs'] / 411:.0f}%** of 411 GB/s ({100 * sm['bf16']['measured_GBs'] / st['bf16']['bandwidth']['copy_rw_GBs']:.0f}% of this session's copy measurement of {st['bf16']['bandwidth']['copy_rw_GBs']:.0f} GB/s). Prefetch variants change it by {pp_worst:+.1f} to {pp_best:+.1f} points relative to no prefetch (limit +10). int8 / int4 reach {100 * sm['int8']['measured_GBs'] / 411:.0f}% / {100 * sm['int4']['measured_GBs'] / 411:.0f}% (instruction-bound dequantization in my kernels). |")
    A(f"| **H63** prediction: local megakernel makes P4 >= 6% faster (bf16 local) and >= 12% (all-int4) | **Refuted** with the measured kernel and with the byte extrapolation; **only the ideal-overlap bound reaches the lines** | near-lossless P4: **{nl_m['gain_pct']:+.1f}%** from the measured step ({nl_m['pred_ms_per_byte']:.3f} vs {nl_m['today_ms_per_byte']:.3f} ms/byte), {nl_i['gain_pct']:+.1f}% at ideal overlap. All-int4 P4: **{i4_e['gain_pct']:+.1f}%** from the step extrapolated by bytes ({i4_e['pred_ms_per_byte']:.3f} vs {i4_e['today_ms_per_byte']:.3f}), {i4_i['gain_pct']:+.1f}% at ideal overlap; my measured (instruction-bound) int4 kernel would be {i4_m['gain_pct']:+.1f}%. If the saving per round halves for K > 0 rounds (they are 99% of P4 rounds) the gains are {nl_m['gain_low_pct']:+.1f}% and {i4_e['gain_low_pct']:+.1f}%. |")
    n_in = sum([1.2 <= dr_bf["ratio_equal_precision"] <= 1.4, 1.2 <= dr_i8["ratio_equal_precision"] <= 1.4, 1.0 <= dr_bf["ratio_local_int4"] <= 1.2, 1.0 <= dr_i8["ratio_local_int4"] <= 1.2])
    A(f"| **H64** floor ratio 1.2-1.4x (equal precision), 1.0-1.2x (Bolmo local int4) | **Not supported as stated** (narrow misses; depends on loop and precision) | drafted loops, equal precision: bf16 **{dr_bf['ratio_equal_precision']:.2f}x**, int8 {dr_i8['ratio_equal_precision']:.2f}x; with Bolmo local int4: bf16 {dr_bf['ratio_local_int4']:.2f}x, int8 {dr_i8['ratio_local_int4']:.2f}x ({n_in} of 4 cells in range). Minimal loops: {mi_bf['ratio_equal_precision']:.2f}x / {mi_i8['ratio_equal_precision']:.2f}x / {mi_i4['ratio_equal_precision']:.2f}x (bf16 / int8 / int4), local int4: {mi_bf['ratio_local_int4']:.2f}x / {mi_i8['ratio_local_int4']:.2f}x. |")
    A("")
    A("The pre-registered **predictions for the record**: *most of the overhead sits inside short kernels* - right (85% inside), but for a different reason than the theory: the short kernels are the ~240 tiny non-matmul kernels, not the ramp of the matmuls. *Counter sync costs 1-2 us per phase* - right at the lower end (0.98 / 1.25 us). *A prefetching persistent kernel reaches 75-85% of copy bandwidth* - right for bf16 (75-78% of 411 GB/s, 78-82% of this session's copy measurement), but so does the same fused phases as 28 separate kernels in a CUDA graph (below), so persistence and prefetch add nothing in my implementation. *A real megakernel would make P4 6-11% faster in bf16 and 14-16% in int4* - **wrong**: 4.3% and 7.5% (7.8% and 12.1% as upper bounds). *Byte tax 1.3x at equal precision and 1.05-1.15x with int4 local* - partly: 1.23x (bf16 drafted) but 1.42x in int8, and 0.97-1.24x with int4 local.")
    A("")
    A("### 2. Stage 0: census of today's local step")
    A("")
    A("Traces: `tools/nsys_rounds.py` (the graphed local pair A_K + C_K, 30 back-to-back replays after 300 warm-up replays, Nsight Systems with `--cuda-graph-trace=node`), analysed by `tools/mk_census.py`. bf16 = skinny Triton kernels, bf16 local weights (config `configs/skinny/bolmo-bf16sk-p4.toml`, K = 0 and the P4 operating point K = 6); all-int4 = torchao tinygemm (`configs/p4/int4-p4.toml`, K = 0 and K = 4), the best int4 path today. Floor = local weight bytes / 411 GB/s: 360.35 MB (bf16, from the real parameter counts: 5 layers x 34.15 M + two 2048 x 2048 projections + `lm_head` 1.06 M = 180.2 M parameters) -> 877 us; 95.7 MB int4 (4-bit + scale and zero per group of 128) -> 233 us. mLSTM state traffic (10.5 MB per round, 25 us) is not in the floor.")
    A("")
    A("| round | kernels | kernel sum us | traced period us | floor us | excess us (share of round) | inside kernels us (share of excess) | between kernels us | kernels < 10 us (time) | matmul kernels (time) | other kernels (time) |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, b, c in (("bf16 skinny K = 0", b0, c0), ("bf16 skinny K = 6", b6, c6), ("int4 tinygemm K = 0", d0, c4), ("int4 tinygemm K = 4", d4, c4)):
        A(f"| {name} | {b['kernels']} | {b['kernel_sum_us']:.0f} | {b['period_us']:.0f} | {c['floor_us']:.0f} | {b['excess_us']:.0f} ({100 * b['share_above_floor']:.0f}%) | {b['inside_excess_us']:.0f} ({100 * b['share_of_excess_inside']:.0f}%) | {b['between_us']:.0f} | {b['n_lt10']} ({b['us_lt10']:.0f} us) | {b['matmul_kernels']} ({b['matmul_us']:.0f} us) | {b['other_kernels']} ({b['other_us']:.0f} us) |")
    A("")
    A(f"Reading. (1) The traced period is 3% above the untraced benchmark of the same pair (bf16 K = 0 {b0['period_us']:.0f} vs 1347 us; int4 {d0['period_us']:.0f} vs 791 us). (2) In bf16 the 42 skinny matmul kernels together take {b0['matmul_us']:.0f} us, **{100 * (b0['matmul_us'] / c0['floor_us'] - 1):.0f}% above the weight-read floor**: the matmuls themselves are nearly at the floor. The excess of {b0['excess_us']:.0f} us is {b0['other_kernels']} other kernels ({b0['other_us']:.0f} us: elementwise, norms/reductions, the mLSTM step, copies; median duration {b0['dur_p50']:.1f} us, so mostly at the ~1 us graph-node floor measured before), plus {b0['between_us']:.0f} us of gaps. (3) In int4 the matmuls are {d0['matmul_us']:.0f} us against a {c4['floor_us']:.0f} us floor ({d0['matmul_us'] - c4['floor_us']:.0f} us over) and the other {d0['other_kernels']} kernels take {d0['other_us']:.0f} us, gaps {d0['between_us']:.0f} us; here the excess is 72% of the round. (4) At the P4 operating points the matmul kernels grow with K (bf16 K = 6: {b6['matmul_us']:.0f} us for {b6['matmul_kernels']} matmul kernels; int4 K = 4: {d4['matmul_us']:.0f} us for {d4['matmul_kernels']}) while the other kernels stay at about {b6['other_us']:.0f} / {d4['other_us']:.0f} us. Splitting the excess as the task asks: **inside kernels** = matmul above floor ({mm_above:.0f} us bf16, {d0['matmul_us'] - c4['floor_us']:.0f} us int4) + other kernels ({b0['other_us']:.0f} / {d0['other_us']:.0f} us); **between kernels** = {b0['between_us']:.0f} / {d0['between_us']:.0f} us. So the overhead is a **kernel-count problem (fusion)**, not a barrier/ramp problem of long kernels.")
    A("")
    A("### 3. Stage 1: synchronization microbenchmark")
    A("")
    A(f"One persistent block per SM ({sync['grid']} blocks x {sync['threads']} threads, residency checked), N dependent phases, each: every thread stores one float, `__syncthreads`, thread 0 does a release-add (`red.release.gpu`), the next phase waits (`ld.acquire.gpu` spin, capped). Total launch time was measured with CUDA events (median of 25 launches x 3 rounds), then a straight line was fitted through N = 10, 20, 40, 80: the slope is the cost per phase, the intercept the launch cost (~45 us for the persistent kernels, ~30 us for cooperative / graph launches). Work-only (no sync) is flat at ~58 us (slope {rows['none (work only)']['us_per_phase_slope']:.3f}).")
    A("")
    A("| variant | N = 10 us | N = 20 | N = 40 | N = 80 | **slope us per phase** |")
    A("|---|---:|---:|---:|---:|---:|")
    order = ["barrier (one counter), +__threadfence", "barrier (one counter)", "barrier, relaxed poll + fence", "barrier, sync only (no read-back)", "dataflow 1 producer", "dataflow 2 producers",
             "dataflow 4 producers", "dataflow all blocks (flags)", "dataflow 1 producer, sync only", "dataflow 2 producers, sync only", "dataflow 4 producers, sync only", "grid.sync()", "graph of tiny kernels"]
    label = {"barrier (one counter), +__threadfence": "(a) barrier, one counter, extra `__threadfence`", "barrier (one counter)": "(a) barrier, one counter (with read-back of a producer value)",
             "barrier, relaxed poll + fence": "(a) barrier, relaxed polling + one fence", "barrier, sync only (no read-back)": "**(a) barrier, sync only**",
             "dataflow 1 producer": "(b) dataflow, 1 producer (with read-back)", "dataflow 2 producers": "(b) dataflow, 2 producers (with read-back)", "dataflow 4 producers": "(b) dataflow, 4 producers (with read-back)",
             "dataflow all blocks (flags)": "(b) all 38 flags polled (with read-back)", "dataflow 1 producer, sync only": "**(b) dataflow, 1 producer, sync only**", "dataflow 2 producers, sync only": "**(b) dataflow, 2 producers, sync only**",
             "dataflow 4 producers, sync only": "**(b) dataflow, 4 producers, sync only**", "grid.sync()": "(c) cooperative `grid.sync()` (cooperative launch supported on this card)", "graph of tiny kernels": "(d) graph of tiny kernels (one node per phase)"}
    for k in order:
        r = rows[k]
        A(f"| {label[k]} | " + " | ".join(f"{r['total_us'][str(n)]:.1f}" for n in (10, 20, 40, 80)) + f" | **{r['us_per_phase_slope']:.2f}** |")
    A("")
    A(f"Reading. The cheapest counter barrier costs {bar:.2f} us per phase, close to cooperative groups ({gs:.2f}) and to one node of a CUDA graph ({gn:.2f}; the earlier fair-ratio task measured ~1.0 us for a trivial node). The release/acquire round trip through L2 sets the cost, not the number of arrivals: letting each block wait for only 1, 2 or 4 producers instead of all 38 saves {100 * (1 - df1 / bar):.0f}%, {100 * (1 - df2 / bar):.0f}% and {100 * (1 - df4 / bar):.0f}%, far from 30%. A redundant `__threadfence` before the release costs 0.1-0.15 us per phase, relaxed polling + fence is slower than `ld.acquire`. Reading a value from a producer after the wait (what any consumer must do) adds another 0.2-0.3 us. **Synchronization is not the problem: 27 barriers cost about 27 us per step.** The unmeasured part is the arrival skew: in Stage 2 the slowest block of a matvec phase finishes {ts['matvec_tail_us']:.1f} us after the mean (median over phases), which a barrier turns into idle time for everyone.")
    A("")
    A("### 4. Stage 2: streaming microbenchmark")
    A("")
    A(f"28 phases, the local stack's real shapes, one row of activations, random weights; step time = median over 7 rounds x 5 trials x 20 CUDA-graph replays (variants interleaved). Weight GB/s = weight bytes / step time. Reference bandwidths in this session (256 MiB buffers): copy (read + write counted) {st['bf16']['bandwidth']['copy_rw_GBs']:.0f} / {st['int8']['bandwidth']['copy_rw_GBs']:.0f} / {st['int4']['bandwidth']['copy_rw_GBs']:.0f} GB/s and read-only {st['bf16']['bandwidth']['read_only_GBs']:.0f} GB/s; the percentage columns use the task's 411 GB/s. The asymptotic bandwidth of the matvec loop itself (one phase, 100-200 MB) is {asym['fit_GBs']:.0f} GB/s (`results/megakernel/asym.json`); a single phase kernel has a fixed cost of {asym['fit_fixed_us']:.1f} us (launch + ramp + tail).")
    A("")
    for q, title in (("bf16", "bf16 (360.35 MB)"), ("int8", "int8, per-row scale (180.51 MB)"), ("int4", "int4, group 128, fp16 scale (92.94 MB)")):
        A(f"**{title}**")
        A("")
        A("| variant | step us | weight GB/s | % of 411 GB/s |")
        A("|---|---:|---:|---:|")
        for k, v in st[q]["variants"].items():
            A(f"| {k} | {v['us_median']:.0f} | {v['weight_GBs']:.0f} | {100 * v['weight_GBs'] / 411:.0f}% |")
        A("")
    A("Variants: (a) one fused phase kernel per phase in a CUDA graph (28 launches; this is already 'perfectly fused' - today's real step has ~280 kernels); (b) persistent, one block per SM, waits for the dependency before loading; (c) persistent, at the end of a phase each block stages the first rows of its next-phase slice (up to 80 KiB) in shared memory with `cp.async` and preloads the next mLSTM state slice, then waits at the barrier; (c2) the same, issued at the start of the previous phase into a double buffer; (c0) state preload only. All variants issue the first batch of weight loads before the input vector is read. For context, the existing Triton skinny linears alone take 0.93 ms (bf16) / 0.59 (int8) / 0.40 (int4 tinygemm) per K = 0 round, plus ~280 small kernels.")
    A("")
    A(f"Correctness: the persistent kernel (without prefetch, both prefetch modes) and the per-phase kernels agree with the fp32 torch reference of the same math to a maximum relative error of about 1e-6 (bf16 1.3e-6, int8 1.4e-6, int4 3.0e-6 on logits; limit in the tests 1e-4), for one step and for two steps (state carried, n / m double buffer swapped, counter base advanced) and after capture in a CUDA graph. Reading: **the persistent kernel is not faster than the 28-kernel graph** ({sm['bf16']['measured_persistent_us']:.0f} vs {sm['bf16']['measured_graph_us']:.0f} us in bf16; {st['int8']['variants']['b: persistent, no prefetch']['us_median']:.0f} vs {sm['int8']['measured_graph_us']:.0f} in int8, {st['int4']['variants']['b: persistent, no prefetch']['us_median']:.0f} vs {sm['int4']['measured_graph_us']:.0f} in int4), and **prefetching the next phase's weights does not help** (the staged bytes just move into the barrier wait; per phase the DRAM time is the same). More blocks per kernel do not help the per-phase variant either. In int8 and int4 my kernels are instruction-bound (several shift / mask / convert instructions per weight, one block per SM): they reach only {100 * sm['int8']['measured_GBs'] / 411:.0f}% and {100 * sm['int4']['measured_GBs'] / 411:.0f}% of copy bandwidth, and more blocks per SM make them slower, so the int8/int4 step times below are properties of my dequantization code, not of the card (the Triton int8 skinny kernel reached 77-91% in the previous task).")
    A("")
    A(f"Where the bf16 step goes (timestamps per block and phase, `results/megakernel/tsprof-bf16.json`, variant b): the ~22 matvec phases need {sm['bf16']['stream_floor_us']:.0f} us at the asymptotic {asym['fit_GBs']:.0f} GB/s; the measured step is {sm['bf16']['measured_persistent_us']:.0f} us. The difference is made of 27 barriers x {bar:.2f} us = {27 * bar:.0f} us, five mLSTM phases of {ts['mlstm_phase_us']:.0f} us each ({5 * ts['mlstm_phase_us']:.0f} us: a dependent chain of state load, update, reduction and store), and a per-phase loss of ~5-8 us (arrival skew {ts['matvec_tail_us']:.1f} us + input-vector load and norm + first-load latency) x 22 matvec phases. The single-kernel fixed cost fitted in `asym.json` ({asym['fit_fixed_us']:.1f} us) is the same loss. A persistent kernel pays it as well; I did not find a way to hide it.")
    A("")
    A("### 5. Stage 3: cost model, predicted ms/byte, floor ratio, gate")
    A("")
    A("**Model.** `pred ms/byte = measured ms/byte - rounds per byte x saving per round`, with the measured cells of `## Skinny kernels` (default prompts) and `saving = today's K = 0 pair (A + C, untraced benchmark) - (megakernel step + extras)`. Extras = 3 small phases the benchmark does not contain (embedding gather and add, argmax / token write, boundary bookkeeping), "
      f"{sm['extra_us']:.0f} us. The saving is applied unchanged to every round, including K > 0 rounds (99% of P4 rounds): this assumes that a tensor-core M > 1 megakernel streams the weights once and adds the same marginal per-row cost as today's fused verify path. **That is not measured here** (Stage 2 is M = 1 only); the 'halved for K > 0' column shows the sensitivity. The host loop (`outside replays`, drafter host time, prefill) stays as measured.")
    A("")
    A("| local step (bf16 / int8 / int4) | weight MB | 28-kernel graph us | persistent best us | stream floor at asymptote us | ideal overlap us | extrapolated by bytes us |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for q in ("bf16", "int8", "int4"):
        m = sm[q]
        ex = f"{m['extrapolated_us']:.0f}" if "extrapolated_us" in m else "(measured)"
        A(f"| {q} | {m['weight_bytes'] / 1e6:.1f} | {m['measured_graph_us']:.0f} | {m['measured_persistent_us']:.0f} | {m['stream_floor_us']:.0f} | {m['ideal_overlap_us']:.0f} | {ex} |")
    A("")
    A(f"*Measured persistent*: the best persistent variant of Stage 2. *Ideal overlap*: stream floor at {asym['fit_GBs']:.0f} GB/s + 27 x {bar:.2f} us sync + 5 mLSTM phases x {ts['mlstm_phase_us']:.0f} us, i.e. the theory's promise that all ramp/tail is hidden; no variant reached it. *Extrapolated by bytes* (int8, int4 only): stream floor + the bf16 step's loss over its own floor ({sm['fixed_us_bf16']:.0f} us), i.e. an int8/int4 kernel that is bandwidth-bound like bf16 is.")
    A("")
    for name, p, key in (("Near-lossless: global int8, local bf16 (skinny)", nl, "bf16"), ("All-int4 (tinygemm, today's best int4)", i4, "int4")):
        A(f"**{name}.** Today's K = 0 local round {p['today_pair_k0_ms']:.3f} ms.")
        A("")
        A("| megakernel step | step + extras ms | saving per round ms | minimal loop ms/byte (today -> pred, gain) | P4 ms/byte (today -> pred, gain) | P4 gain if the saving halves for K > 0 |")
        A("|---|---:|---:|---:|---:|---:|")
        for lab in ("measured kernel", "extrapolated by bytes", "ideal overlap"):
            if lab == "extrapolated by bytes" and key == "bf16":
                continue
            r = p[lab]
            A(f"| {lab} | {r['mega_k0_ms']:.3f} | {r['delta_per_round_ms']:+.3f} | {r['minimal']['today_ms_per_byte']:.3f} -> {r['minimal']['pred_ms_per_byte']:.3f} ({r['minimal']['gain_pct']:+.1f}%) | {r['P4']['today_ms_per_byte']:.3f} -> {r['P4']['pred_ms_per_byte']:.3f} ({r['P4']['gain_pct']:+.1f}%) | {r['P4']['gain_low_pct']:+.1f}% |")
        A("")
    A("**Fair floor ratio (H64).** Both models at the weight-read floor, bytes per generated byte / 411 GB/s, from the real parameter counts (Bolmo local 180.17 M, Bolmo global 1.0737 G, OLMo 2 blocks 1.0737 G, OLMo 2 head 0.2097 G; int4 = 0.53125 byte per parameter) and the measured loop statistics: minimal loop = 1 / 3.38 global passes per byte and 1 local round per byte for Bolmo, 0.275 passes per byte for OLMo; drafted = P4 (bytes per global pass 7.4 bf16 / 6.6 int8, rounds per byte 0.328 / 0.352) against OLMo combined K = 8 (0.130 / 0.125 passes per byte). Bolmo's floor includes 10.5 MB of mLSTM state traffic per round; OLMo's KV cache is not counted. Because both floors scale with the same achieved-bandwidth fraction, the ratio of 'equal megakernel efficiency' is the ratio of floors.")
    A("")
    A("| loop | precision | Bolmo GB/byte | OLMo 2 GB/byte | **ratio, equal precision** | **ratio, Bolmo local int4** | (OLMo head bf16: equal / local int4) | Bolmo floor ms/byte | OLMo floor ms/byte |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in fr["rows"]:
        A(f"| {r['loop']} | {r['precision']} | {r['bolmo_GB_per_byte']:.3f} | {r['olmo_GB_per_byte']:.3f} | **{r['ratio_equal_precision']:.2f}x** | **{r['ratio_local_int4']:.2f}x** | {r['ratio_equal_precision_head_bf16']:.2f}x / {r['ratio_local_int4_head_bf16']:.2f}x | {r['bolmo_floor_ms_per_byte']:.3f} | {r['olmo_floor_ms_per_byte']:.3f} |")
    A("")
    A(f"Measured today: Bolmo g8 P4 1.588 vs OLMo 2 int8 skinny K = 8 0.994 ms/byte = **1.60x** (`## Skinny kernels`), against a floor ratio of {dr_i8['ratio_equal_precision']:.2f}x in int8; in bf16 1.36x against {dr_bf['ratio_equal_precision']:.2f}x. Both models are far from their floors (Bolmo {100 * dr_i8['bolmo_floor_ms_per_byte'] / 1.588:.0f}% and OLMo {100 * dr_i8['olmo_floor_ms_per_byte'] / 0.994:.0f}% of floor speed in int8); the weight-independent costs (norms, attention, cache, host loop) dominate both. The 'with local int4' columns are not a deployable setting by themselves (a global int8 + local int4 model has a different quality; see `## Local-stack quantization`).")
    A("")
    A("**Gate (nothing is built here).** Recommend building a real local megakernel only if (i) the predicted P4 gain is >= 10% in either precision, (ii) sync <= 2 us per phase, (iii) streaming >= 75% of copy bandwidth.")
    A("")
    A("| condition | value | result |")
    A("|---|---|---|")
    A(f"| (i) predicted P4 gain >= 10% | near-lossless {g_nl['pred_gain_pct_measured_kernel']:+.1f}% (measured step), {g_nl['pred_gain_pct_ideal']:+.1f}% (ideal overlap); all-int4 {g_i4['pred_gain_pct_extrapolated']:+.1f}% (bytes extrapolation), {g_i4['pred_gain_pct_ideal']:+.1f}% (ideal overlap) | **fails** (best realistic {max(g_nl['pred_gain_pct_measured_kernel'], g_i4['pred_gain_pct_extrapolated']):.1f}%; even the ideal-overlap bound reaches 10% only in int4) |")
    A(f"| (ii) sync <= 2 us per phase | {bar:.2f} us (sync only), {bar_full:.2f} us with read-back | passes |")
    A(f"| (iii) streaming >= 75% of copy bandwidth | bf16 {100 * sm['bf16']['measured_GBs'] / 411:.0f}%, int8 {100 * sm['int8']['measured_GBs'] / 411:.0f}%, int4 {100 * sm['int4']['measured_GBs'] / 411:.0f}% | bf16 passes, int8 / int4 fail (my dequantization is instruction-bound) |")
    A("")
    A(f"**Recommendation: do not build the persistent local megakernel now.** Condition (i) fails: the near-lossless variant needs a saving of about {need_delta:.0f} us per round to reach 10% of 1.588 ms/byte at 0.352 rounds/byte, the model predicts {nl['measured kernel']['delta_per_round_ms'] * 1000:.0f} us (measured step) to {nl['ideal overlap']['delta_per_round_ms'] * 1000:.0f} us (ideal). What would change it: (1) a persistent kernel that actually hides the per-phase loss (~5-8 us x 22 phases), which would need either register-level cross-barrier prefetch with finer dynamic work distribution against the arrival skew, or a hardware feature I did not use (e.g. thread-block clusters are not available on sm_86); (2) an int4 dequantization that reaches the bandwidth (4-bit weights pay off only then); (3) an M > 1 (K > 0) megakernel, since 99% of P4 rounds verify 7+ rows. What the data does support is the cheaper cousin: **fuse the ~240 small non-matmul kernels** of the local round (their 371-397 us and the gaps are the whole excess), e.g. as the 28 fused phase kernels modelled here in a CUDA graph (1.13 ms against 1.35-1.39 ms today in bf16: a 0.2 ms saving per round without any persistence or spin-waiting, if the real round fuses as well as the model).")
    A("")
    A("### 6. Engineering problem or hardware floor?")
    A("")
    A(f"Mostly a hardware floor, with one engineering-sized slice. At the bandwidth floor Bolmo still reads {dr_bf['ratio_equal_precision']:.2f}x (bf16) to {dr_i8['ratio_equal_precision']:.2f}x (int8) the bytes of OLMo 2 per generated byte, because its local stack is read once per round (0.33-0.35 rounds per byte) next to the global pass (0.14-0.15 per byte); no kernel can remove those reads on a card whose caches hold 4 MB. The measured byte tax of 1.60x (int8) sits only about 13% above that floor ratio, so a perfect local megakernel could not bring it below ~1.5x; my model predicts {nl_m['gain_pct']:.1f}-{nl_i['gain_pct']:.1f}% of Bolmo's time, which moves the ratio from 1.60x to about {1.60 * (1 - nl_m['gain_pct'] / 100):.2f}-{1.60 * (1 - nl_i['gain_pct'] / 100):.2f}x. The remaining distance to the floor is large in both models (each runs at ~35-40% of floor speed) and is made of weight-independent work, not of local-stack launch overhead: in the local round it is ~240 tiny kernels (fusable without a megakernel), in the global pass ~2 ms of norms/attention/cache per pass, in the host loop the sync. So the byte tax is a floor in what it reads, and an engineering problem in how long everything else takes; the persistent-megakernel mechanism of the theory (hide barriers and ramps) did not materialise on this card, because a barrier is 1 us and the loss that remains per phase is arrival skew and dependent-chain latency, which prefetching weights cannot hide.")
    A("")
    A("### 7. Deviations, problems and uncertainties")
    A("")
    A("- **The Stage 2 stack is a model of the local round, not the round.** Random weights; simplified mLSTM math; no convolution, no per-head norm, no boundary logic, no embeddings (3 extra phases are added as an allowance of "
      f"{sm['extra_us']:.0f} us); M = 1 only. The saving per round for K > 0 rounds is extrapolated (see the model paragraph). Numerical correctness is checked against a torch reference of the same simplified math, not against Bolmo.")
    A("- **Variant (a) is not today's real graph.** Task wording: 'today's style: one kernel per phase in a CUDA graph (skinny kernels)'. I used my own 28 fused phase kernels (the best case of that style) and took the real baseline from Stage 0 and the earlier benchmarks (1.347 ms bf16 g8 pair, 0.791 ms int4 pair). The Triton skinny linears were not re-benchmarked inside the chain.")
    A("- **Dataflow waits (b) were measured in Stage 1 only.** In the matvec chain every phase needs the whole input vector, so the only head-local dependency is A -> B; I used full barriers in Stage 2. With 1-4 producers saving <= 9% per phase (Stage 1), this cannot change the result.")
    A("- **int4 was implemented, not skipped** (group-128 int4 with fp16 scales, tested against the reference) but it is instruction-bound at 23% of copy bandwidth; the int8 and int4 step times in the cost model are therefore an extrapolation by bytes plus the bf16 per-phase loss, as the task allows, and the measured values are shown next to them. A faster dequantizer (packed half2 math, `lop3` tricks) was not tried.")
    A("- **Prefetch implementations tried**: end-of-phase `cp.async` staging (80 / 32 KiB), double-buffered staging at the start of the previous phase (80 KiB), mLSTM state preload, first-batch weight loads before the input vector. Not tried: register-level prefetch across the barrier, dynamic row stealing against the arrival skew, `cp.async.bulk` / TMA (not on sm_86), L2 prefetch hints. H62's second part is refuted for these implementations; a better one might exist.")
    A("- **Nsight Systems**: the bf16 skinny trace with NVTX ranges could not be imported twice (`AnalysisFailed`/`InvalidArgument`, the int4 trace with the same tool imported fine). I took one trace per K without NVTX (`run_megakernel_stage0c.sh`; the last 30 of 330 identical rounds are analysed). The traced period is ~3% above the untraced benchmark. The int4 K = 0 / K = 4 traces and the bf16 ones are single traces, one prompt state (the graphs are replayed on whatever state is in the buffers).")
    A(f"- **Bandwidth reference.** The task's 411 GB/s was used for floors and percentages; this session's own copy measurements were {st['bf16']['bandwidth']['copy_rw_GBs']:.0f}-{st['int8']['bandwidth']['copy_rw_GBs']:.0f} GB/s (read + write counted) and read-only {st['bf16']['bandwidth']['read_only_GBs']:.0f} GB/s; GPU clocks drift and were not locked. Step-time differences below ~3% between variants are within noise (the 28-kernel graph and the persistent kernel are the same within it).")
    A(f"- **Bugs found on the way** (all fixed, covered by tests): int4 `xsum` was computed per group of 128 but subtracted per vector of 32 (wrong results); the group-sum loop covered only 8 of 16 groups; the barrier base used 28 instead of 27 barriers per step, so the second launch timed out (caught by the error flag, no hang); mLSTM n / m were read and written by different blocks in the same phase (race) and are now double-buffered by step parity. {sum(m['killed'] for m in mut)} of {len(mut)} injected mutants are killed by `tests/test_megakernel_bench.py` (`tools/mk_mutation.py`, `results/megakernel/mutation.json`); the spin-cap code itself is not mutation-tested because a mutant that ignores the cap would hang the display driver.")
    A("- **Overlap.** One CPU-only analysis of the int4 trace (sqlite) ran while the second Nsight trace was being recorded; the full test suite was started only after all measurements. Nothing else used the GPU during a measurement.")
    A("- **Tests.** `python -m pytest -q`: 355 passed (310 before this task + 45 in `tests/test_megakernel_bench.py`: reference comparison for the three precisions, persistent / per-phase / graph, two-step state carry, publication check, spin-cap on missing producers in both the sync and the streaming kernel, refusal of non-resident grids, a corrupted-weights check).")
    A("- The Colab notebook was not updated; nothing was committed to git.")
    A("")
    text = "\r\n".join(L) + "\r\n"
    old = F.read_text(encoding="utf-8") if False else open(F, encoding="utf-8", newline="").read()
    snap = open(SNAP, encoding="utf-8", newline="").read()
    assert old.startswith(snap), "FINDINGS.md prefix changed"
    assert "### Environment and method" not in old[len(snap):], "results already appended"
    open(F, "w", encoding="utf-8", newline="").write(old + text)
    new = open(F, encoding="utf-8", newline="").read()
    assert new.startswith(open(ROOT / "results/FINDINGS.before_megakernel.md", encoding="utf-8", newline="").read())
    print("appended", len(text), "chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
