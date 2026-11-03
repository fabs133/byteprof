"""Task B, Stage B0: render results/crossing/b0/sim_*.json (+ cost_*.json) as results/crossing/b0/sim.md."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/crossing/b0"
KS = (4, 8, 12, 16, 24, 32)
NAMES = ("smollm2-135m", "smollm2-360m", "qwen2.5-0.5b")


def load(name):
    p = OUT / f"sim_{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(h, q):
    items = sorted((int(k), v) for k, v in h.items())
    tot = sum(v for _, v in items)
    acc = 0
    for k, v in items:
        acc += v
        if acc >= q * tot:
            return k
    return 0


def main() -> int:
    L = ["# B0.3: emitted bytes per round under the multi-patch rule (extended set, 256 new bytes, pooled)", "",
         "Round = one local verification pass; emitted = 1 + accepted draft bytes (byte and flag must match, boundaries",
         "do not end the round, at most 8 patches per round). Flags from dolma2 re-tokenization (`cut` = the task's",
         "protocol: cut to K bytes, tokenize context + draft, drop the last token; `full` = tokenize context + the",
         "whole 12-token continuation, drop only its last token, then cut to K). `raw` / `healed` = drafter context.", ""]
    data = {n: load(n) for n in NAMES}
    base = load("combined")
    hdr = "| drafter | context | flags | " + " | ".join(f"K={k}" for k in KS) + " |"
    sep = "|---|---|---|" + "---:|" * len(KS)
    L += ["## Multi-patch bytes per round", "", hdr, sep]
    for n, d in data.items():
        if not d:
            continue
        for cm in ("raw", "healed"):
            for fm in ("cut", "full"):
                L.append(f"| {n} ({d['params'] / 1e6:.0f}M) | {cm} | {fm} | " + " | ".join(
                    f"{d['cells'][f'{cm}|{fm}|{k}|multi']['ext']['bytes_per_round']:.2f}" for k in KS) + " |")
    if base:
        for fm in ("cut",):
            L.append(f"| combined (context + n-gram), flags re-tokenized | - | {fm} | " + " | ".join(
                f"{base['cells'][f'{fm}|{k}|multi']['ext']['bytes_per_round']:.2f}" for k in KS) + " |")
    L += ["", "## Within-patch rule with the same continuations (bytes per round; for the gain over within-patch drafting)", "", hdr, sep]
    for n, d in data.items():
        if not d:
            continue
        for cm in ("raw", "healed"):
            for fm in ("cut", "full"):
                L.append(f"| {n} | {cm} | {fm} | " + " | ".join(
                    f"{d['cells'][f'{cm}|{fm}|{k}|single']['ext']['bytes_per_round']:.2f}" for k in KS) + " |")
    if base:
        L.append("| combined, flags re-tokenized | - | cut | " + " | ".join(
            f"{base['cells'][f'cut|{k}|single']['ext']['bytes_per_round']:.2f}" for k in KS) + " |")
    L += ["", "## Bytes per global pass (multi-patch), share of rounds that cross at least one boundary (`healed`, `full`)", "",
          "| drafter | " + " | ".join(f"K={k}" for k in KS) + " |", "|---|" + "---:|" * len(KS)]
    for n, d in data.items():
        if not d:
            continue
        L.append(f"| {n} bytes per global pass | " + " | ".join(
            f"{d['cells'][f'healed|full|{k}|multi']['ext']['bytes_per_global']:.2f}" for k in KS) + " |")
        L.append(f"| {n} crossing share | " + " | ".join(
            f"{d['cells'][f'healed|full|{k}|multi']['ext']['crossing_share']:.3f}" for k in KS) + " |")
    if base:
        L.append("| combined bytes per global pass | " + " | ".join(
            f"{base['cells'][f'cut|{k}|multi']['ext']['bytes_per_global']:.2f}" for k in KS) + " |")
        L.append("| combined crossing share | " + " | ".join(
            f"{base['cells'][f'cut|{k}|multi']['ext']['crossing_share']:.3f}" for k in KS) + " |")
    L += ["", "## By prompt type, best cell of each drafter (multi-patch, K = 32, `healed`, `full`)", "",
          "| drafter | " + " | ".join(("prose_en", "code_py", "prose_de", "json", "unicode_mix")) + " |", "|---|---:|---:|---:|---:|---:|"]
    for n, d in data.items():
        if not d:
            continue
        bt = d["cells"]["healed|full|32|multi"]["by_type"]
        L.append(f"| {n} | " + " | ".join(f"{bt[t]:.2f}" for t in ("prose_en", "code_py", "prose_de", "json", "unicode_mix")) + " |")
    L += ["", "## Default prompts (5, reported only; `healed`, `full`, multi-patch)", "", hdr.replace("context | flags", "context | flags"), sep]
    for n, d in data.items():
        if not d:
            continue
        L.append(f"| {n} | healed | full | " + " | ".join(
            f"{d['cells'][f'healed|full|{k}|multi']['default']['bytes_per_round']:.2f}" for k in KS) + " |")
    L += ["", "## OLMo 2: tokens per round, token-level acceptance, the same continuations (extended set, 72 tokens ~ 256 bytes; K in tokens = K/4)", "",
          "| drafter | context | " + " | ".join(f"K={k}B ({k // 4} tok)" for k in KS) + " |", "|---|---|" + "---:|" * len(KS)]
    for n, d in data.items():
        if not d:
            continue
        for cm in ("raw", "healed"):
            L.append(f"| {n} | {cm} | " + " | ".join(f"{d['olmo'][f'{cm}|{k}']['tokens_per_round']:.2f}" for k in KS) + " |")
    L += ["", "Same, in emitted **bytes** per round (tokens x bytes per token of the emitted text):", "",
          "| drafter | context | " + " | ".join(f"K={k}" for k in KS) + " |", "|---|---|" + "---:|" * len(KS)]
    for n, d in data.items():
        if not d:
            continue
        for cm in ("raw", "healed"):
            L.append(f"| {n} | {cm} | " + " | ".join(f"{d['olmo'][f'{cm}|{k}']['bytes_per_round']:.2f}" for k in KS) + " |")
    L += ["", "## Drafter tokens needed per round (`healed`, `full`, multi-patch): median / p90 / max", "",
          "| drafter | " + " | ".join(f"K={k}" for k in KS) + " |", "|---|" + "---:|" * len(KS)]
    for n, d in data.items():
        if not d:
            continue
        cells = []
        for k in KS:
            h = d["cells"][f"healed|full|{k}|multi"]["m_hist"]
            cells.append(f"{pct(h, .5)} / {pct(h, .9)} / {max(int(x) for x in h)}")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    costs = {n: json.loads((OUT / f"cost_{n}.json").read_text(encoding="utf-8")) for n in NAMES if (OUT / f"cost_{n}.json").exists()}
    if costs:
        L += ["", "# B0.4: drafter round cost (ms per graph replay: catch-up forward over c tokens, then M-1 single-token steps)", ""]
        for n, c in costs.items():
            ms = sorted({int(k.split(",")[1]) for k in c["ms"]})
            cs = sorted({int(k.split(",")[0]) for k in c["ms"]})
            L += [f"**{n}** ({c['params'] / 1e6:.0f}M parameters, bf16, context {c['ctx_tokens']} tokens, static cache {c['max_cache_len']}, peak VRAM {c['peak_vram_mib']:.0f} MiB)", "",
                  "| c \\ M | " + " | ".join(str(m) for m in ms) + " |", "|---|" + "---:|" * len(ms)]
            for ci in cs:
                L.append(f"| {ci} | " + " | ".join(f"{c['ms'][f'{ci},{m}']:.3f}" for m in ms) + " |")
            L.append("")
    (OUT / "sim.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
