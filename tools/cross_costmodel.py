"""Task B, Stage B1: extended cost model for crossing (multi-patch) drafts, break-even, predictions and the gate.

round cost = drafter (graph of c = 3 catch-up tokens + M - 1 steps, measured in B0.4; M = p90 of the tokens a round needs)
           + local verify pair(K) (Task A fused path, results/parallel/bench/<prec>-fused-L0)
           + global pass B(P) + (C_p - C_n) for the P patches the round runs speculatively (P <= 8, measured here)
           + outside-replays per round + host cost of the drafter (measured: recorded-continuation lookup + dolma2 retokenization)
ms/byte = prefill share + sum(rounds x round cost) / emitted bytes. Terms outside the round as in tools/parallel_costmodel.py.

Usage: python tools/cross_costmodel.py   -> results/crossing/b1/costmodel.{json,md}
"""

import json
import pickle
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from byteprof.drafting.drafters import CombinedDrafter, ContextDrafter, NullDrafter  # noqa: E402
from byteprof.drafting.flags import Dolma2  # noqa: E402
from byteprof.drafting.lmdraft import PrecomputedLMDrafter, build_draft  # noqa: E402
from byteprof.drafting.ngram import NGramDrafter, NGramModel  # noqa: E402
from byteprof.drafting.sim import simulate  # noqa: E402
from byteprof.drafting.truthio import load_truth  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from cross_sim_report import Retokenized  # noqa: E402

B0 = ROOT / "results/crossing/b0"
OUT = ROOT / "results/crossing/b1"
KS_ALL = (1, 2, 4, 6, 8, 12, 16, 24, 32)
KS_EVAL = (4, 8, 12, 16, 24, 32)
NAMES = ("smollm2-135m", "smollm2-360m", "qwen2.5-0.5b")
MAX_PATCHES = 8
C_CATCHUP = 3
OLMO_VERIFY = {0: 9.27, 1: 9.37, 2: 9.40, 4: 9.88, 6: 9.93, 8: 10.01}  # ms per graph, drafting task (K tokens)
K_TOK = {4: 1, 8: 2, 12: 3, 16: 4, 24: 6, 32: 8}
STAGE1 = ROOT / "results/drafting/stage1"


def olmo_verify(k: int) -> float:
    ks = sorted(OLMO_VERIFY)
    for a, b in zip(ks, ks[1:]):
        if a <= k <= b:
            return OLMO_VERIFY[a] + (OLMO_VERIFY[b] - OLMO_VERIFY[a]) * (k - a) / (b - a)
    return OLMO_VERIFY[ks[-1]]


class Timed(PrecomputedLMDrafter):
    times: list[float] = []

    def propose(self, k):
        t = time.perf_counter()
        r = super().propose(k)
        Timed.times.append(time.perf_counter() - t)
        return r


def pct(h: dict, q: float) -> int:
    items = sorted((int(k), v) for k, v in h.items())
    tot = sum(v for _, v in items)
    acc = 0
    for k, v in items:
        acc += v
        if acc >= q * tot:
            return k
    return items[-1][0] if items else 1


def draft_cost(cost: dict, m: int) -> float:
    ms = {int(k.split(",")[1]): v for k, v in cost["ms"].items() if k.startswith(f"{C_CATCHUP},")}
    xs = sorted(ms)
    slope = (ms[xs[-1]] - ms[xs[0]]) / (xs[-1] - xs[0])
    return ms[xs[0]] + slope * (m - xs[0])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dol = Dolma2()
    cases_all = {p: load_truth(ROOT / f"results/drafting/truth/{p}.json") for p in ("bf16", "int4")}
    parallel = json.loads((ROOT / "results/parallel/costmodel.json").read_text(encoding="utf-8"))
    recs = {n: pickle.load(open(B0 / "drafts" / f"{n}.pkl", "rb")) for n in NAMES}
    recs_int4 = {n: pickle.load(open(B0 / "drafts" / f"{n}-int4.pkl", "rb")) for n in NAMES}  # continuations on the int4 truth
    costs = {n: json.loads((B0 / f"cost_{n}.json").read_text(encoding="utf-8")) for n in NAMES}
    corpus = ROOT / "results/drafting/corpus"
    stage0 = json.loads((ROOT / "results/drafting/stage0/stage0.json").read_text(encoding="utf-8"))
    host_comb = stage0["host_cost_us"]["combined|8"]["median_us"]
    ngram = NGramModel.build((corpus / "corpus.bin").read_bytes(), np.fromfile(corpus / "flags.bin", np.uint8),
                             (1, 2, 3, 4, 5, 6, 8, 12), bits=24)
    out: dict = {"inputs": {}, "bolmo": {}, "olmo": {}, "breakeven": {}}
    md = ["# B1: cost model for crossing drafts", ""]

    for prec in ("bf16", "int4"):
        tim = json.loads(sorted((STAGE1 / f"timing-{prec}").glob("*/breakdown.json"))[-1].read_text(encoding="utf-8"))["rows"]
        ins = [r["instr"] for r in tim]
        nbytes = sum(r["new_bytes"] for r in tim)
        prefill_pb = sum(i["prefill_ms"] for i in ins) / nbytes
        o_round = sum(i["outside_replays_ms"] for i in ins) / sum(i["decode_steps"] for i in ins)
        bench = json.loads((ROOT / f"results/parallel/bench/{prec}-fused-L0/bench.json").read_text(encoding="utf-8"))
        pair = {int(k): v["ms"] for k, v in bench["rounds"].items()}
        dp = bench["pieces"]["Cp|0"]["ms"] - bench["pieces"]["Cn|0"]["ms"]
        gm = json.loads((ROOT / f"results/drafting/stage4/global_multi-{prec}-p8.json").read_text(encoding="utf-8"))
        B = {0: 0.0, **{p: gm[f"P={p}"]["ms"] for p in range(1, 9)}}
        out["inputs"][prec] = {"prefill_per_byte": prefill_pb, "outside_per_round_ms": o_round, "pair": pair, "Cp_minus_Cn": dp, "B": B}
        truth = cases_all[prec]
        sets = {"default": [c for c in truth.values() if c.set == "default"], "ext": [c for c in truth.values() if c.set == "drafting_ext"]}

        def predict(sname, make, k, multi, d_cost, host_us, d_floor=None):
            tot = {"E": 0, "R": 0, "kp": {}}
            mh: dict = {}
            for c in sets[sname]:
                d = make(c)
                r = simulate(c.truth[256], d, k, ctx_atoms=c.ctx_atoms, max_new=256, multi_patch=multi,
                             max_patches=MAX_PATCHES if multi else None, ks=[x for x in KS_ALL if x <= k] if k else [0])
                tot["E"] += r.emitted
                tot["R"] += r.rounds
                for key, n in r.by_kp.items():
                    tot["kp"][key] = tot["kp"].get(key, 0) + n
                for m, v in getattr(d, "m_hist", {}).items():
                    mh[m] = mh.get(m, 0) + v
            m_k = pct(mh, 0.9) if mh else 0
            dc = d_cost(m_k) if m_k else 0.0
            loc = sum(n * (pair[kk] + B[p] + (dp if p else 0)) for (kk, p), n in tot["kp"].items())
            ms = prefill_pb + (loc + tot["R"] * (o_round + dc + host_us / 1000)) / tot["E"]
            rpb = tot["R"] / tot["E"]
            extra = {}
            if d_floor is not None:  # sensitivity: drafter at the weight-bandwidth floor, and a free drafter
                extra = {"draft_floor_ms": d_floor(m_k) if m_k else 0.0,
                         "ms_floor": ms - rpb * dc + rpb * (d_floor(m_k) if m_k else 0.0), "ms_free": ms - rpb * dc}
            return {"ms_per_byte": ms, "bytes_per_round": tot["E"] / tot["R"], "m_p90": m_k, "draft_ms": dc,
                    "rounds_per_byte": rpb, **extra}

        # sanity: K = 0 loop of this model against the Task A model
        k0 = {s: predict(s, lambda c: NullDrafter(), 0, False, lambda m: 0.0, 0.0)["ms_per_byte"] for s in sets}
        ref = {s: parallel[prec]["fused"][f"{s}|2"]["k0_ms_per_byte"] for s in sets}
        out["inputs"][prec]["k0"] = {"mine": k0, "task_a": ref}
        # best within-patch drafting so far: Task A's combined drafter with the fused verification
        best_w = {}
        for s in sets:
            kb = max((2, 4, 6, 8, 12, 16, 24, 32), key=lambda kk: parallel[prec]["fused"][f"ext|{kk}"]["gain"])
            best_w[s] = {"k_max": kb, "ms_per_byte": parallel[prec]["fused"][f"{s}|{kb}"]["ms_per_byte"]}
        out["inputs"][prec]["best_within_patch"] = best_w
        res: dict = {}
        for n in NAMES:
            rec = recs[n] if prec == "bf16" else recs_int4[n]
            for cm in ("raw", "healed"):
                for fm in ("cut", "full"):
                    for k in KS_EVAL:
                        for multi in (True, False):
                            Timed.times = []

                            def mk(c, cm=cm, fm=fm):
                                return Timed(rec["bolmo"][c.id][cm], dol, fm)

                            for s in sets:
                                p = predict(s, mk, k, multi, lambda m, n=n: draft_cost(costs[n], m), 0.0,
                                            lambda m, n=n: m * costs[n]['params'] * 2 / 448e9 * 1000)
                                res[f"{n}|{cm}|{fm}|{k}|{'multi' if multi else 'single'}|{s}"] = p
                            host = statistics.median(Timed.times) * 1e6
                            # add the measured host cost of the drafter (the recorded-continuation replay + retokenization)
                            for s in sets:
                                x = res[f"{n}|{cm}|{fm}|{k}|{'multi' if multi else 'single'}|{s}"]
                                x["host_us"] = host
                                for key_ in ("ms_per_byte", "ms_floor", "ms_free"):
                                    x[key_] += x["rounds_per_byte"] * host / 1000
                    print(prec, n, cm, fm, flush=True)
        # baseline: the existing combined drafter with dolma2 flags (cut protocol), multi-patch and within-patch
        for k in KS_EVAL:
            for multi in (True, False):
                for s_ in sets:
                    res[f"combined|-|cut|{k}|{'multi' if multi else 'single'}|{s_}"] = predict(
                        s_, lambda c: Retokenized(CombinedDrafter(ContextDrafter(min_len=3), NGramDrafter(ngram, min_count=1, max_order=6)), dol, "cut"),
                        k, multi, lambda m: 0.0, host_comb + 150.0)
        out["bolmo"][prec] = {"cells": res, "k0": k0}

        # selection on the extended set only, applied to the default prompts (the speed workload)
        def best(filter_):
            cand = [(key, v) for key, v in res.items() if key.endswith("|ext") and filter_(key)]
            return min(cand, key=lambda kv: kv[1]["ms_per_byte"])

        sel = {}
        for label, f in (("multi, any drafter", lambda k_: "|multi|" in k_),
                         ("single, any drafter", lambda k_: "|single|" in k_),
                         ("multi, token LMs", lambda k_: "|multi|" in k_ and not k_.startswith("combined")),
                         ("single, token LMs", lambda k_: "|single|" in k_ and not k_.startswith("combined"))):
            key, v = best(f)
            d_key = key[:-3] + "default"
            sel[label] = {"cell": key[:-4], "ext": v, "default": res[d_key]}
        out["bolmo"][prec]["selected"] = sel
        sens = {}
        for var in ("ms_per_byte", "ms_floor", "ms_free"):
            for rule in ("multi", "single"):
                cand = [(key, v) for key, v in res.items() if key.endswith("|ext") and f"|{rule}|" in key and not key.startswith("combined")]
                key, v = min(cand, key=lambda kv: kv[1][var])
                sens[f"{var}|{rule}"] = {"cell": key[:-4], "ext": v[var], "default": res[key[:-3] + "default"][var],
                                         "bpr": v["bytes_per_round"], "draft_ms": v.get("draft_ms") if var == "ms_per_byte" else (v.get("draft_floor_ms") if var == "ms_floor" else 0.0)}
        out["bolmo"][prec]["sensitivity"] = sens
        md += ["", "Token-LM drafters under three drafter costs (best cell per row chosen on ext): measured HF-eager graph, the weight-bandwidth floor "
               "(parameters x 2 bytes / 448 GB/s per token, no launch cost), and a free drafter (acceptance effect only):", "",
               "| drafter cost | rule | cell | B/round ext | drafter ms/round | ms/byte ext | ms/byte default | vs best within-patch (default) |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for key_, v in sens.items():
            var, rule = key_.split("|")
            md.append(f"| {dict(ms_per_byte='measured graph', ms_floor='bandwidth floor', ms_free='free')[var]} | {rule} | {v['cell']} | {v['bpr']:.2f} | {v['draft_ms']:.1f} | "
                      f"{v['ext']:.3f} | {v['default']:.3f} | {(1 - v['default'] / best_w['default']['ms_per_byte']) * 100:+.1f}% |")
        md += [f"## {prec}", "", f"K = 0 check (this model / Task A model, ms/byte): default {k0['default']:.3f} / {ref['default']:.3f}, "
               f"ext {k0['ext']:.3f} / {ref['ext']:.3f}. Prefill {prefill_pb:.3f} ms/byte, outside-replays {o_round * 1000:.0f} us/round. "
               f"Best within-patch drafting so far (combined drafter, fused verification, k_max chosen on ext): "
               f"ext {best_w['ext']['ms_per_byte']:.3f} (k_max {best_w['ext']['k_max']}), default {best_w['default']['ms_per_byte']:.3f} ms/byte.", "",
               "Best cell by predicted ms/byte on the extended set (selection on ext; the default column applies the cell to the 5 speed prompts):", "",
               "| choice | cell | B/round ext | ms/byte ext | ms/byte default | vs best within-patch (default) |", "|---|---|---:|---:|---:|---:|"]
        for label, v in sel.items():
            md.append(f"| {label} | {v['cell']} | {v['ext']['bytes_per_round']:.2f} | {v['ext']['ms_per_byte']:.3f} | "
                      f"{v['default']['ms_per_byte']:.3f} | {(1 - v['default']['ms_per_byte'] / best_w['default']['ms_per_byte']) * 100:+.1f}% |")
        md += ["", "All cells, extended set (ms/byte; drafter graph ms and bytes per round in brackets; `healed`/`full` for the LMs):", "",
               "| drafter | rule | " + " | ".join(f"K={k}" for k in KS_EVAL) + " |", "|---|---|" + "---:|" * len(KS_EVAL)]
        for n in (*NAMES, "combined"):
            for rule in ("multi", "single"):
                cells = []
                for k in KS_EVAL:
                    key = f"{n}|healed|full|{k}|{rule}|ext" if n != "combined" else f"combined|-|cut|{k}|{rule}|ext"
                    v = res[key]
                    cells.append(f"{v['ms_per_byte']:.2f} ({v['draft_ms']:.1f}; {v['bytes_per_round']:.1f})")
                md.append(f"| {n} | {rule} | " + " | ".join(cells) + " |")
        md.append("")

    # ---- OLMo 2 (bf16): token-level acceptance, same drafters
    olmo_truth = json.loads((ROOT / "results/drafting/truth/olmo-bf16.json").read_text(encoding="utf-8"))["prompts"]
    bt = statistics.mean(
        sum(len(dol.id_bytes(t)) for t in o["tokens"]["72"][1:73]) / 72 for o in olmo_truth.values() if o["set"] == "default")
    # calibrate prefill share and per-round host time on the two measured cells (minimal loop 2.681; combined K=8 1.419, 0.130 rounds/byte)
    v0, v8 = olmo_verify(0), olmo_verify(8)
    pf_o, h_o = 2.681 - v0 / bt, 0.0  # K = 0 cell fixes the prefill share; a two-cell fit gave a negative host time
    chk8 = pf_o + 0.130 * v8
    out["olmo"]["calibration"] = {"bytes_per_token": bt, "prefill_per_byte": pf_o, "host_per_round_ms": h_o}
    md += ["## OLMo 2 bf16 (token-level acceptance, the same drafters)", "",
           f"Round = verify graph (K tokens, drafting task: 9.27 ms at K = 0, 10.01 at K = 8) + drafter + host. Prefill share {pf_o:.3f} ms/byte and "
           f"no host time per round, calibrated on the minimal-loop cell (2.681 ms/byte, {bt:.2f} bytes per token); the same model gives {chk8:.3f} for the measured combined K = 8 cell (1.419 measured, 0.130 rounds/byte). "
           "The drafter host cost is taken as the Bolmo value (about 0.1 ms) for the LM cells.", "",
           "| drafter | K (bytes / tokens) | tokens per round | bytes per round | drafter ms/round | predicted ms/byte |", "|---|---|---:|---:|---:|---:|"]
    olmo_pred = {}
    recs_params = {n: costs[n]['params'] for n in NAMES}
    for n in NAMES:
        sim = json.loads((B0 / f"sim_{n}.json").read_text(encoding="utf-8"))["olmo"]
        for k in KS_EVAL:
            c = sim[f"healed|{k}"]
            m = pct(c["m_hist"], 0.9) if c["m_hist"] else 1
            dc = draft_cost(costs[n], m)
            rpb = 1 / c["bytes_per_round"]
            ms = pf_o + rpb * (olmo_verify(K_TOK[k]) + dc + h_o + 0.11)
            fl = m * recs_params[n] * 2 / 448e9 * 1000
            ms_floor = pf_o + rpb * (olmo_verify(K_TOK[k]) + fl + h_o + 0.11)
            olmo_pred[f"{n}|{k}"] = {"ms_per_byte": ms, "ms_floor": ms_floor, "bytes_per_round": c["bytes_per_round"], "tokens_per_round": c["tokens_per_round"], "draft_ms": dc}
            md.append(f"| {n} | {k} / {K_TOK[k]} | {c['tokens_per_round']:.2f} | {c['bytes_per_round']:.2f} | {dc:.1f} | {ms:.3f} |")
    out["olmo"]["pred"] = olmo_pred
    md.append("")
    best_olmo_lm = min(olmo_pred.values(), key=lambda v: v["ms_per_byte"])["ms_per_byte"]
    sel = out["bolmo"]["bf16"]["selected"]
    md += ["## Byte tax (bf16, speed workload = default prompts)", "",
           "| Bolmo | OLMo 2 | ratio |", "|---|---|---:|",
           f"| minimal loop {out['bolmo']['bf16']['k0']['default']:.3f} (measured 3.995) | minimal loop 2.688 (measured) | {3.995 / 2.688:.2f} |",
           f"| within-patch combined fused {out['inputs']['bf16']['best_within_patch']['default']['ms_per_byte']:.3f} (Task A measured 3.330) | combined K = 8 1.420 (measured) | {3.330 / 1.420:.2f} |",
           f"| best predicted with a token-LM drafter, multi-patch {sel['multi, token LMs']['default']['ms_per_byte']:.3f} | best predicted OLMo with a token-LM drafter {best_olmo_lm:.3f} | "
           f"{sel['multi, token LMs']['default']['ms_per_byte'] / best_olmo_lm:.2f} |", ""]

    # ---- break-even (H35): drafter free, accuracy-limited; within-patch emits min(b, bytes per patch)
    bpp = 3.79
    md += ["## Break-even (H35)", "",
           "Model: a drafter that costs nothing and emits `b` bytes per round. Within-patch drafting emits min(b, bytes per patch) per round and runs the global model "
           "once per patch; multi-patch emits b per round and runs the global pass for the P = 1 + K/bytes-per-patch patches it drafts, every round. "
           f"Bytes per patch = {bpp} (extended set, no drafting). Break-even b* = smallest b with multi <= within (per byte).", "",
           "| precision | K | P | b* (bytes/round) | b* / bytes per patch |", "|---|---:|---:|---:|---:|"]
    for prec in ("bf16", "int4"):
        I = out["inputs"][prec]
        pair, B, dp, o = I["pair"], I["B"], I["Cp_minus_Cn"], I["outside_per_round_ms"]
        for k in (8, 16, 32):
            P = min(8, 1 + int(round(k / bpp)))
            found = None
            for b100 in range(100, 2000):
                b = b100 / 100
                multi = (pair[k] + B[P] + dp + o) / b
                within = (pair[k] + o) / min(b, bpp) + (B[1] + dp) / bpp
                if multi <= within:
                    found = b
                    break
            out["breakeven"][f"{prec}|{k}"] = {"b_star": found, "ratio": found / bpp if found else None, "P": P}
            md.append(f"| {prec} | {k} | {P} | {found} | {found / bpp:.2f} |" if found else f"| {prec} | {k} | {P} | none below 20 | - |")
    md.append("")
    (OUT / "costmodel.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (OUT / "costmodel.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
