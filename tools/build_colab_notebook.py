"""Builds colab_profile.ipynb from plain cell sources.

The notebook is generated so its code stays reviewable as plain Python.
Edit this file, then run:  python tools/build_colab_notebook.py
"""
import json
import sys
from pathlib import Path

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# byteprof on Colab: Bolmo 1B vs. OLMo 2 1B

Profiles the byte-level **Bolmo 1B** and its token-level source model **OLMo 2 1B**
with the same harness, prompts, seed and dtype, on a Linux GPU. Linux matters
for two reasons: the torch profiler records CUDA kernel events here (it didn't
on Windows), and there is no WDDM launch-latency penalty.

**Before running**
1. Locally, Claude Code has finished `TASK_byte_count.md` (byte-count fix + `inspect` command).
2. The project folder is on Google Drive **without** `.venv/` and `results/`.
   Set `DRIVE_REPO` below to its path.
3. Runtime → Change runtime type → GPU. An **L4 or A100** is preferred (native bf16).
   A T4 works but runs both models in fp32 (fp16 makes Bolmo overflow).

**What you get:** `results/colab/<tag>/summary.md` on Drive, plus every
`breakdown.md` / `breakdown.json` / `kernels.txt`. Bring `summary.md` and the
breakdowns back to the chat.
""")

code(r"""
# --- Settings ---------------------------------------------------------------
DRIVE_REPO = "/content/drive/MyDrive/Model_Research/Byte_Based_Input"  # uploaded project folder
WORK = "/content/Byte_Based_Input"          # local working copy (fast disk)
MODELS = {                                  # name -> config file (relative to the project)
    "bolmo-1b": "configs/bolmo-1b.toml",
    "olmo2-1b": "configs/olmo2-1b.toml",
}
VARIANTS = {}                # optional optimized variants, name -> config file; empty = skipped. E.g.
                             # {"olmo2-1b-compile": "configs/olmo2-1b-compile.toml",
                             #  "bolmo-1b-graph": "configs/bolmo-1b-graph.toml",
                             #  "bolmo-1b-graph-global": "configs/bolmo-1b-graph-global.toml",
                             #  "bolmo-1b-minimal": "configs/bolmo-1b-minimal.toml",
                             #  "olmo2-1b-compile-minimal": "configs/olmo2-1b-compile-minimal.toml"}
FULL_RUNS = 2                # full profiles per model; compared for noise
DTYPE_OVERRIDE = None        # None = auto (bf16 on Ampere+, fp32 otherwise); or "bfloat16"/"float16"/"float32"
CACHE_HF_ON_DRIVE = False    # True: keep model downloads on Drive across sessions
                             # (saves re-downloading, but reading from Drive is slower)
""")

code(r"""
# --- GPU, dtype and environment ----------------------------------------------
import platform, subprocess, sys, torch

assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > GPU, then rerun."
GPU = torch.cuda.get_device_name(0)
CAP = torch.cuda.get_device_capability(0)
# bf16 needs Ampere or newer. Without it, use fp32: Bolmo overflows in fp16 on a T4
# (it generates only special tokens, which inspect reports as new_bytes = 0).
DTYPE = DTYPE_OVERRIDE or ("bfloat16" if CAP >= (8, 0) else "float32")
print(f"GPU: {GPU}  compute capability {CAP}  -> dtype {DTYPE}")
if DTYPE != "bfloat16":
    print(f"WARNING: running both models in {DTYPE}. Absolute numbers are not comparable "
          "with bf16 runs; the Bolmo/OLMo ratio and the GPU busy fraction still are.")
print("python", platform.python_version(), "| torch", torch.__version__, "| CUDA", torch.version.cuda)
!nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv
!lscpu | grep -E "Model name|^CPU\(s\)"
""")

code(r"""
# --- Mount Drive and make a local working copy --------------------------------
from google.colab import drive
drive.mount("/content/drive")
import os
assert os.path.isdir(DRIVE_REPO), f"Not found: {DRIVE_REPO} - fix DRIVE_REPO in the settings cell."
!rsync -a --delete --exclude .venv --exclude results --exclude __pycache__ --exclude .pytest_cache "{DRIVE_REPO}/" "{WORK}/"
%cd {WORK}
!ls
""")

code(r"""
# --- Install dependencies -------------------------------------------------------
# Keep Colab's preinstalled CUDA build of torch; skip torch and Windows-only lines.
!grep -vE "^\s*(#|$)" requirements.txt | grep -viE "^(torch|triton-windows)([<>=;! ]|$)" > /tmp/req-colab.txt
!echo "--- installing:"; cat /tmp/req-colab.txt
!pip install -q -r /tmp/req-colab.txt
!python -c "import transformers; print('transformers', transformers.__version__)"
""")

code(r"""
# --- Helpers ---------------------------------------------------------------------
import os, subprocess, sys

os.environ["HF_HOME"] = (os.path.join(os.path.dirname(DRIVE_REPO), "hf_cache")
                         if CACHE_HF_ON_DRIVE else "/content/hf_cache")
print("HF_HOME =", os.environ["HF_HOME"])

def run(*args, module="byteprof"):
    # Runs `python -m <module> ...`, streams its output, raises on failure.
    cmd = [sys.executable, "-m", module, *map(str, args)]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    code = proc.wait()
    if code:
        raise RuntimeError(f"exit code {code}: {' '.join(cmd)}")
""")

code(r"""
# --- Tests ---------------------------------------------------------------------
run("-q", module="pytest")
""")

code(r"""
# --- Byte-count check (inspect) ---------------------------------------------------
# Needs the `inspect` command from TASK_byte_count.md. It exits non-zero if any
# prompt generates more bytes than the cap allows or the prompt is miscounted.
import json, time

probe = subprocess.run([sys.executable, "-m", "byteprof", "inspect", "--help"],
                       capture_output=True, text=True)
assert probe.returncode == 0, ("byteprof has no `inspect` command. Run TASK_byte_count.md "
                               "in Claude Code locally first, then re-upload the folder.")

TAG = time.strftime("%Y%m%d-%H%M%S") + "-" + GPU.replace(" ", "_")
BASE = f"results/colab/{TAG}"
for name, cfg in MODELS.items():
    run("inspect", "--config", cfg, "--dtype", DTYPE, "--output-dir", f"{BASE}/{name}/inspect")
    rows = json.load(open(f"{BASE}/{name}/inspect/inspect.json"))
    per_run = sum(r["new_bytes"] for r in rows) / len(rows)
    print(f"==> {name}: {per_run:.0f} new bytes per generation on average\n")
""")

md(r"""
If the two averages above differ a lot, adjust `max_new_bytes` in
`configs/olmo2-1b.toml` (it counts **tokens** for OLMo) here in the working copy,
then rerun the cell above. Similar output lengths keep the KV-cache sizes comparable.
""")

code(r"""
# --- Profiling runs ------------------------------------------------------------------
for name, cfg in MODELS.items():
    out = f"{BASE}/{name}"
    common = ["--config", cfg, "--dtype", DTYPE]
    run("profile", *common, "--output-dir", f"{out}/smoke", "--max-new-bytes", 32, "--runs", 1, "--warmup", 1)
    for _ in range(FULL_RUNS):
        run("profile", *common, "--output-dir", f"{out}/full")
    run("profile", *common, "--output-dir", f"{out}/trace", "--runs", 1, "--trace")
""")

code(r'''
# --- Analysis: throughput, component shares, GPU busy fraction -----------------------
import glob, json

def merged_busy_us(intervals):
    """Total length of the union of (start, end) intervals."""
    busy, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    return busy

assert merged_busy_us([(0, 10), (5, 15), (20, 25)]) == 20  # tiny self-check

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}

def gpu_stats(trace_path):
    """GPU busy fraction over the whole traced generate(), kernel count, mean kernel time."""
    events = json.load(open(trace_path)).get("traceEvents", [])
    timed = [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]
    gpu = [(e["ts"], e["ts"] + e["dur"]) for e in timed if e.get("cat") in GPU_CATS]
    kernels = [e["dur"] for e in timed if e.get("cat") == "kernel"]
    if not timed or not gpu:
        return None
    window = max(e["ts"] + e["dur"] for e in timed) - min(e["ts"] for e in timed)
    return {"busy": merged_busy_us(gpu) / window, "kernels": len(kernels),
            "mean_kernel_us": sum(kernels) / len(kernels) if kernels else 0.0}

def summarize_full(paths):
    rows = []
    for p in paths:
        d = json.load(open(p))
        dec = d["decode"]
        base_ms = d["derived"]["baseline_wall_ms"]
        rows.append({
            "bytes_per_run": d["generated_bytes"] / d["runs"],
            "bps_nohooks": d["generated_bytes"] / (base_ms / 1000) if base_ms else float("nan"),
            "decode_ms_per_byte": d["derived"]["decode_ms_per_byte"],
            "shares": {k: v["total_ms"] / dec["step_ms"] for k, v in dec["components"].items()} if dec["step_ms"] else {},
            "unattributed": d["derived"]["decode_unattributed_ms"] / dec["step_ms"] if dec["step_ms"] else float("nan"),
            "warnings": d["warnings"],
        })
    return rows

lines = [f"# Colab profile {TAG}", "", f"- GPU: {GPU}, dtype {DTYPE}, torch {torch.__version__}", ""]
per_model = {}
for name in MODELS:
    fulls = sorted(glob.glob(f"{BASE}/{name}/full/*/breakdown.json"))
    traces = sorted(glob.glob(f"{BASE}/{name}/trace/*/trace.json"))
    rows = summarize_full(fulls)
    per_model[name] = rows
    lines += [f"## {name}", ""]
    for i, r in enumerate(rows, 1):
        shares = ", ".join(f"{k} {100*v:.1f}%" for k, v in sorted(r["shares"].items(), key=lambda kv: -kv[1]))
        lines.append(f"- full run {i}: {r['decode_ms_per_byte']:.3f} ms/byte decode, "
                     f"{r['bps_nohooks']:.1f} bytes/s (no hooks), {r['bytes_per_run']:.0f} bytes/run")
        lines.append(f"  - decode shares: {shares}, unattributed {100*r['unattributed']:.1f}%")
        if r["warnings"]:
            lines.append(f"  - warnings: {len(r['warnings'])} (see breakdown.md)")
    if len(rows) >= 2:
        a, b = rows[0]["decode_ms_per_byte"], rows[1]["decode_ms_per_byte"]
        lines.append(f"- run-to-run difference: {100*abs(a-b)/min(a,b):.1f}%")
    stats = gpu_stats(traces[-1]) if traces else None
    if stats:
        lines.append(f"- trace: GPU busy {100*stats['busy']:.1f}% of generate() wall time, "
                     f"{stats['kernels']} kernels, mean kernel {stats['mean_kernel_us']:.1f} us")
    else:
        lines.append("- trace: no GPU events found")
    lines.append("")

if all(per_model.get(n) for n in ("bolmo-1b", "olmo2-1b")):
    bo = sum(r["decode_ms_per_byte"] for r in per_model["bolmo-1b"]) / len(per_model["bolmo-1b"])
    ol = sum(r["decode_ms_per_byte"] for r in per_model["olmo2-1b"]) / len(per_model["olmo2-1b"])
    lines += ["## Comparison", "", f"- Bolmo / OLMo 2 decode ms per generated byte: {bo/ol:.2f}x "
              f"({bo:.3f} vs {ol:.3f})", ""]

summary = "\n".join(lines)
open(f"{BASE}/summary.md", "w").write(summary)
print(summary)
''')

code(r"""
# --- Optional: optimized variants (skipped while VARIANTS is empty) ---------------------
# CUDA graphs bypass forward hooks, so these runs report wall time only (no component
# breakdown) plus first divergence from eager output. The eager `full` runs above are the
# reference for speedups.
for name, cfg in VARIANTS.items():
    out = f"{BASE}/{name}"
    common = ["--config", cfg, "--dtype", DTYPE]
    for _ in range(FULL_RUNS):
        run("profile", *common, "--output-dir", f"{out}/full")
    run("profile", *common, "--output-dir", f"{out}/trace", "--runs", 1, "--trace")
""")

code(r'''
# --- Optional: summary of optimized variants (appended to summary.md) ---------------------
if VARIANTS:
    def eager_ms_per_byte(model):
        vals = []
        for p in sorted(glob.glob(f"{BASE}/{model}/full/*/breakdown.json")):
            d = json.load(open(p))
            if d["derived"]["baseline_wall_ms"]:
                vals.append(d["derived"]["baseline_wall_ms"] / d["generated_bytes"])
        return sum(vals) / len(vals) if vals else float("nan")

    vl = ["## Optimized variants (wall time / new bytes, no hooks; eager reference = no-hook wall time / bytes)", ""]
    for name in VARIANTS:
        base = "bolmo-1b" if name.startswith("bolmo") else "olmo2-1b"
        for i, p in enumerate(sorted(glob.glob(f"{BASE}/{name}/full/*/breakdown.json")), 1):
            d = json.load(open(p))
            tot_b = sum(r["new_bytes"] for r in d["rows"])
            tot_ms = sum(r["ms"] for r in d["rows"])
            divs = {}
            for r in d["rows"]:
                divs.setdefault(r["prompt_id"], set()).add(r["first_divergence"])
            ref = eager_ms_per_byte(base)
            vl.append(f"- {name} run {i}: {tot_ms/tot_b:.3f} ms/byte, {1000*tot_b/tot_ms:.1f} bytes/s, "
                      f"speedup vs eager {ref/(tot_ms/tot_b):.2f}x; first divergence per prompt: "
                      + ", ".join(f"{k}={'-' if v == {None} else sorted(x for x in v if x is not None)}" for k, v in divs.items()))
            vl.append(f"  - {d['meta']['optimization']['details']}")
        traces = sorted(glob.glob(f"{BASE}/{name}/trace/*/trace.json"))
        stats = gpu_stats(traces[-1]) if traces else None
        if stats:
            vl.append(f"  - trace: GPU busy {100*stats['busy']:.1f}%, {stats['kernels']} kernels, "
                      f"mean kernel {stats['mean_kernel_us']:.1f} us")
    vtext = "\n".join(vl)
    open(f"{BASE}/summary.md", "a").write("\n" + vtext + "\n")
    print(vtext)
''')

code(r'''
# --- Optional: GPU idle time split into launch gaps and sync bubbles ----------------------
# Uses byteprof.tracegaps.analyze_gaps (pure, tested): gaps between consecutive GPU kernels;
# a gap is a sync bubble if a cudaStreamSynchronize / cudaDeviceSynchronize / cudaEventSynchronize /
# cudaMemcpy* call overlaps it, otherwise a launch gap. Needs CUDA events in trace.json (Linux).
import sys
sys.path.insert(0, WORK)
from byteprof.tracegaps import analyze_gaps

gl = ["## GPU idle time: launch gaps vs sync bubbles (one traced generation each)", "",
      "| run | idle % | sync-bubble % | launch-gap % | gap median us | gap p90 us | kernels |", "|---|---:|---:|---:|---:|---:|---:|"]
for name in list(MODELS) + list(VARIANTS):
    traces = sorted(glob.glob(f"{BASE}/{name}/trace/*/trace.json"))
    if not traces:
        continue
    g = analyze_gaps(json.load(open(traces[-1])).get("traceEvents", []))
    if g is None:
        gl.append(f"| {name} | no GPU kernel events | | | | | |")
        continue
    gl.append(f"| {name} | {100*g['idle_frac']:.1f} | {100*g['sync_bubble_frac']:.1f} | {100*g['launch_gap_frac']:.1f} "
              f"| {g['gap_median_us']:.1f} | {g['gap_p90_us']:.1f} | {g['kernels']} |")
gtext = chr(10).join(gl)
open(f"{BASE}/summary.md", "a").write(chr(10) + gtext + chr(10))
print(gtext)
''')

code(r"""
# --- Save results to Drive --------------------------------------------------------
dest = f"{DRIVE_REPO}/results/colab"
!mkdir -p "{dest}"
!cp -r "{BASE}" "{dest}/"
print("Saved to", f"{dest}/{TAG}")
""")

md(r"""
## Reading the summary

- **Bolmo / OLMo 2 ratio:** how much more each generated byte costs in the byte
  model. Close to 1.0 means the byte architecture isn't the bottleneck; HF eager
  decoding is.
- **GPU busy %:** low (well under ~50%) with a mean kernel time of a few
  microseconds means the GPU is mostly waiting for kernel launches. That is the
  case where CUDA graphs or `torch.compile(mode="reduce-overhead")` pay off.
- **Compare with the Windows numbers** in `FINDINGS.md`: if the Linux decode
  cost is much lower, WDDM launch latency was a big part of what we measured.
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "notebooks" / "colab" / "colab_profile.ipynb"
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
print("wrote", out)
