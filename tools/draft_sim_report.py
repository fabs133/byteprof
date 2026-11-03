"""Stage 0, steps 3-5: simulate every drafter x K x prompt type x length on the recorded ground truth.

Usage: python tools/draft_sim_report.py --truth results/drafting/truth/bf16.json
Drafter parameters are tuned on the extended prompt set only (25 prompts, 256 new bytes, K = 4); the default
prompts (used for the speed table) are only reported. Output: results/drafting/stage0/{report.md,stage0.json}
"""

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from byteprof.drafting.drafters import CombinedDrafter, ContextDrafter, NullDrafter, OracleDrafter  # noqa: E402
from byteprof.drafting.ngram import NGramDrafter, NGramModel  # noqa: E402
from byteprof.drafting.sim import simulate  # noqa: E402
from byteprof.drafting.truthio import Case, load_truth, repetition_rate  # noqa: E402

KS = (0, 1, 2, 4, 6, 8)
TYPES = ("prose_en", "code_py", "prose_de", "json", "unicode_mix")
CORPUS = ROOT / "results" / "drafting" / "corpus"
ORDERS = (1, 2, 3, 4, 5, 6, 8, 12)


def load_ngram() -> NGramModel:
    data = (CORPUS / "corpus.bin").read_bytes()
    flags = np.fromfile(CORPUS / "flags.bin", np.uint8)
    return NGramModel.build(data, flags, ORDERS, bits=24)


class Pool:
    """Pooled results of one cell over several prompts."""

    def __init__(self) -> None:
        self.emitted = self.rounds = self.global_passes = self.drafted = self.accepted = 0
        self.l_hist: dict[int, int] = {}
        self.per_prompt: list[float] = []

    def add(self, r) -> None:
        self.emitted += r.emitted
        self.rounds += r.rounds
        self.global_passes += r.global_passes
        self.drafted += r.drafted
        self.accepted += r.accepted
        for l, n in r.l_hist.items():
            self.l_hist[l] = self.l_hist.get(l, 0) + n
        self.per_prompt.append(r.bytes_per_local_pass)

    @property
    def bpl(self) -> float:
        return self.emitted / self.rounds if self.rounds else 0.0

    @property
    def bpg(self) -> float:
        return self.emitted / self.global_passes if self.global_passes else float("inf")

    def dump(self) -> dict:
        return {"bytes_per_local_pass": self.bpl, "bytes_per_global_pass": self.bpg, "rounds": self.rounds,
                "emitted": self.emitted, "global_passes": self.global_passes, "drafted": self.drafted,
                "accepted": self.accepted, "l_hist": self.l_hist,
                "macro_bytes_per_local_pass": statistics.mean(self.per_prompt) if self.per_prompt else 0.0}


def run_cell(cases: list[Case], make, k: int, n: int, multi: bool = False) -> Pool:
    pool = Pool()
    for c in cases:
        truth = c.truth.get(n)
        if not truth or len(truth) < 2:
            continue
        d = make()
        pool.add(simulate(truth, d, k, ctx_atoms=c.ctx_atoms, max_new=n, multi_patch=multi))
    return pool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="results/drafting/truth/bf16.json")
    ap.add_argument("--out", default="results/drafting/stage0")
    a = ap.parse_args()
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_truth(ROOT / a.truth)
    ext = [c for c in cases.values() if c.set == "drafting_ext"]
    dflt = [c for c in cases.values() if c.set == "default"]
    t0 = time.perf_counter()
    model = load_ngram()
    t_build = time.perf_counter() - t0
    print(f"ngram built in {t_build:.1f}s", flush=True)
    res: dict = {"ngram_build_s": t_build, "n_ext": len(ext), "n_default": len(dflt)}

    # ---- tuning on the extended set (256 new bytes, K = 4) -------------------------------
    def ctx(m):
        return lambda: ContextDrafter(min_len=m)

    def ng(mc, mo):
        return lambda: NGramDrafter(model, min_count=mc, max_order=mo)

    def comb(m, mc, mo):
        return lambda: CombinedDrafter(ContextDrafter(min_len=m), NGramDrafter(model, min_count=mc, max_order=mo))

    tune: dict = {"context": {}, "ngram": {}, "combined": {}}
    for m in (3, 4, 5, 6, 8):
        tune["context"][str(m)] = run_cell(ext, ctx(m), 4, 256).bpl
    for mc, mo in itertools.product((1, 2, 3, 5, 10), (3, 4, 5, 6, 8, 12)):
        tune["ngram"][f"{mc},{mo}"] = run_cell(ext, ng(mc, mo), 4, 256).bpl
    for m, mc, mo in itertools.product((3, 4, 5, 6, 8), (1, 2, 3, 5), (4, 6, 8, 12)):
        tune["combined"][f"{m},{mc},{mo}"] = run_cell(ext, comb(m, mc, mo), 4, 256).bpl
    best_ctx = int(max(tune["context"], key=tune["context"].get))
    best_ng = tuple(int(x) for x in max(tune["ngram"], key=tune["ngram"].get).split(","))
    best_cb = tuple(int(x) for x in max(tune["combined"], key=tune["combined"].get).split(","))
    res["tune"] = tune
    res["chosen"] = {"context_min_len": best_ctx, "ngram_min_count": best_ng[0], "ngram_max_order": best_ng[1],
                     "combined": {"min_len": best_cb[0], "min_count": best_cb[1], "max_order": best_cb[2]}}
    print("chosen", res["chosen"], flush=True)

    drafters = {
        "context": ctx(best_ctx),
        "ngram": ng(*best_ng),
        "combined": comb(*best_cb),
        "oracle (upper bound)": lambda: OracleDrafter(),
    }
    sets = {"ext": ext, "default": dflt}
    grid: dict = {}
    for sname, cs in sets.items():
        for n in (256, 1024):
            for dname, make in drafters.items():
                for k in KS:
                    grid[f"{sname}|{n}|{dname}|{k}|all"] = run_cell(cs, make, k, n).dump()
                    if sname == "ext" and dname in ("combined", "oracle (upper bound)", "context", "ngram"):
                        for t in TYPES:
                            grid[f"{sname}|{n}|{dname}|{k}|{t}"] = run_cell([c for c in cs if c.type == t], make, k, n).dump()
    res["grid"] = grid

    # ---- repetition rates -----------------------------------------------------------------
    rep: dict = {}
    for sname, cs in sets.items():
        for n in (256, 1024):
            by: dict[str, list[int]] = {}
            for c in cs:
                tr = c.truth.get(n, [])
                if len(tr) < 2:
                    continue
                prompt = c.prompt_bytes + bytes([tr[0] & 255])
                out = bytes(x & 255 for x in tr[1:])
                r, t = repetition_rate(prompt, out)
                for key in ("all", c.type):
                    by.setdefault(key, [0, 0])
                    by[key][0] += r
                    by[key][1] += t
            rep[f"{sname}|{n}"] = {k: v[0] / v[1] for k, v in by.items()}
    res["repetition"] = rep

    # ---- host cost per round ----------------------------------------------------------------
    class Timed:
        def __init__(self, d):
            self.d, self.t = d, []
            self._p = 0

        def reset(self, ctx_atoms, truth=None):
            self.d.reset(ctx_atoms, truth)

        def propose(self, k):
            t0 = time.perf_counter_ns()
            r = self.d.propose(k)
            self._p = time.perf_counter_ns() - t0
            return r

        def push(self, atoms):
            t0 = time.perf_counter_ns()
            self.d.push(atoms)
            self.t.append(self._p + time.perf_counter_ns() - t0)

    host: dict = {}
    for dname in ("context", "ngram", "combined"):
        for k in (1, 4, 8):
            ts: list[int] = []
            for c in ext:
                d = Timed(drafters[dname]())
                simulate(c.truth[256], d, k, ctx_atoms=c.ctx_atoms, max_new=256)
                ts += d.t
            us = sorted(x / 1000 for x in ts)
            host[f"{dname}|{k}"] = {"median_us": statistics.median(us), "p90_us": us[int(0.9 * (len(us) - 1))],
                                    "rounds": len(us)}
    res["host_cost_us"] = host

    # ---- multi-patch simulation (Stage 4 gate / H28) ----------------------------------------
    mp: dict = {}
    for dname in ("combined", "context", "oracle (upper bound)"):
        for k in KS[1:]:
            s = run_cell(ext, drafters[dname], k, 256, multi=False)
            m = run_cell(ext, drafters[dname], k, 256, multi=True)
            mp[f"{dname}|{k}"] = {"single": s.dump(), "multi": m.dump()}
    base = run_cell(ext, lambda: NullDrafter(), 0, 256)
    res["multi_patch"] = mp
    res["bytes_per_patch_k0_ext256"] = base.bpg
    (out_dir / "stage0.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    # ---- markdown -----------------------------------------------------------------------------
    L = []
    w = L.append
    ch = res["chosen"]
    w("# Stage 0: offline simulation\n")
    w(f"Ground truth: `{a.truth}` ({len(ext)} extended + {len(dflt)} default prompts). n-gram build {t_build:.1f} s. "
      f"Chosen on the extended set (256 new bytes, K = 4, pooled emitted bytes per local pass): context min_len "
      f"{ch['context_min_len']}; n-gram min_count {ch['ngram_min_count']}, max_order {ch['ngram_max_order']}; "
      f"combined min_len {ch['combined']['min_len']}, min_count {ch['combined']['min_count']}, "
      f"max_order {ch['combined']['max_order']}.\n")

    def table(sname, n, key="bytes_per_local_pass", types=("all",)):
        w(f"| drafter | prompts | " + " | ".join(f"K={k}" for k in KS) + " |")
        w("|---|---|" + "---:|" * len(KS))
        for dname in drafters:
            for t in types:
                cells = []
                for k in KS:
                    g = grid.get(f"{sname}|{n}|{dname}|{k}|{t}")
                    cells.append(f"{g[key]:.3f}" if g else "-")
                w(f"| {dname} | {t} | " + " | ".join(cells) + " |")
        w("")

    for sname in ("ext", "default"):
        for n in (256, 1024):
            w(f"## Emitted bytes per local pass, {sname} prompts, {n} new bytes (pooled)\n")
            table(sname, n)
    for n in (256, 1024):
        w(f"## By prompt type (extended set, {n} new bytes), emitted bytes per local pass\n")
        w("| drafter | type | " + " | ".join(f"K={k}" for k in KS) + " |")
        w("|---|---|" + "---:|" * len(KS))
        for dname in ("context", "ngram", "combined", "oracle (upper bound)"):
            for t in TYPES:
                w(f"| {dname} | {t} | " + " | ".join(f"{grid[f'ext|{n}|{dname}|{k}|{t}']['bytes_per_local_pass']:.3f}" for k in KS) + " |")
        w("")
    w("## Repetition rate of the ground-truth outputs (share of 8-byte windows seen earlier in prompt + output)\n")
    w("| set | new bytes | all | " + " | ".join(TYPES) + " |")
    w("|---|---:|---:|" + "---:|" * len(TYPES))
    for key, v in rep.items():
        s, n = key.split("|")
        w(f"| {s} | {n} | {v['all']:.3f} | " + " | ".join(f"{v.get(t, float('nan')):.3f}" for t in TYPES) + " |")
    w("")
    w("## Bytes per global pass (single-patch rounds), extended set, 256 new bytes\n")
    w(f"K = 0 (bytes per patch): {res['bytes_per_patch_k0_ext256']:.3f}\n")
    w("## Multi-patch simulation (extended set, 256 new bytes): emitted bytes per local pass and per global pass\n")
    w("| drafter | K | single: B/local | single: B/global | multi: B/local | multi: B/global | multi/single B per global | multi B/global vs bytes per patch |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, v in mp.items():
        dname, k = key.split("|")
        s, m = v["single"], v["multi"]
        w(f"| {dname} | {k} | {s['bytes_per_local_pass']:.3f} | {s['bytes_per_global_pass']:.3f} | {m['bytes_per_local_pass']:.3f} | "
          f"{m['bytes_per_global_pass']:.3f} | {m['bytes_per_global_pass'] / s['bytes_per_global_pass']:.3f} | "
          f"{m['bytes_per_global_pass'] / res['bytes_per_patch_k0_ext256']:.3f} |")
    w("")
    w("## Drafter host cost per round (propose + push, extended set, 256 new bytes)\n")
    w("| drafter | K | median us | p90 us | rounds |")
    w("|---|---:|---:|---:|---:|")
    for key, v in host.items():
        d, k = key.split("|")
        w(f"| {d} | {k} | {v['median_us']:.1f} | {v['p90_us']:.1f} | {v['rounds']} |")
    w("")
    w("## Distribution of accepted drafts L (combined, extended set, 256 new bytes)\n")
    for k in (4, 8):
        h = grid[f"ext|256|combined|{k}|all"]["l_hist"]
        tot = sum(h.values())
        w(f"K={k}: " + ", ".join(f"L={l}: {n / tot:.3f}" for l, n in sorted(h.items(), key=lambda x: int(x[0]))) + "\n")
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", out_dir / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
