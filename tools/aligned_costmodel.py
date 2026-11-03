"""Patch-aligned task, steps 2 and 5: cost model for mixed round types, old vs new (healed2) tables, byte tax.

Usage: python tools/aligned_costmodel.py   (needs results/aligned/tallies_<prec>_<name>.json from tools/aligned_sim.py)
Output: results/aligned/costmodel.{json,md}

round cost = verify pair(K)  [Task A fused path, results/parallel/bench/<prec>-fused-L0]
           + B(P) + (C_p - C_n) when the round runs P >= 1 patches through the global model   [P=1 for a within-patch round
             that starts at a boundary, 1 + flagged draft atoms for a crossing round]
           + outside replays per round + lookup host cost (combined drafter, +150 us for the retokenization that gives
             the lookup draft its flags) + per LM call: the drafter's graph (measured / bandwidth floor / free) + the
             replay-and-retokenize host time.
ms/byte = prefill share + sum(round costs) / emitted bytes. Terms and inputs as tools/cross_costmodel.py.
"""

import json
import pickle
import statistics
import sys
from itertools import accumulate
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from byteprof.drafting.flags import Dolma2  # noqa: E402
from byteprof.drafting.lmdraft import PrecomputedTokenDrafter  # noqa: E402
from byteprof.drafting.tokens import simulate_tokens  # noqa: E402
from byteprof.drafting.truthio import load_truth  # noqa: E402

OUT = ROOT / "results/aligned"
B0 = ROOT / "results/crossing/b0"
STAGE1 = ROOT / "results/drafting/stage1"
NAMES = ("smollm2-135m", "smollm2-360m", "qwen2.5-0.5b")
KS_EVAL = (4, 8, 12, 16, 24, 32)
SHOW_K = (8, 16, 32)
K_TOK = {4: 1, 8: 2, 12: 3, 16: 4, 24: 6, 32: 8}
OLMO_VERIFY = {0: 9.27, 1: 9.37, 2: 9.40, 4: 9.88, 6: 9.93, 8: 10.01}
RETOK_US = 150.0
C_CATCHUP = 3
COSTS = ("measured", "floor", "free")


def olmo_verify(k: int) -> float:
    ks = sorted(OLMO_VERIFY)
    for a, b in zip(ks, ks[1:]):
        if a <= k <= b:
            return OLMO_VERIFY[a] + (OLMO_VERIFY[b] - OLMO_VERIFY[a]) * (k - a) / (b - a)
    return OLMO_VERIFY[ks[-1]]


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


class Inputs:
    def __init__(self, prec: str):
        tim = json.loads(sorted((STAGE1 / f"timing-{prec}").glob("*/breakdown.json"))[-1].read_text(encoding="utf-8"))["rows"]
        ins = [r["instr"] for r in tim]
        nbytes = sum(r["new_bytes"] for r in tim)
        self.prefill_pb = sum(i["prefill_ms"] for i in ins) / nbytes
        self.o_round = sum(i["outside_replays_ms"] for i in ins) / sum(i["decode_steps"] for i in ins)
        bench = json.loads((ROOT / f"results/parallel/bench/{prec}-fused-L0/bench.json").read_text(encoding="utf-8"))
        self.pair = {int(k): v["ms"] for k, v in bench["rounds"].items()}
        self.dp = bench["pieces"]["Cp|0"]["ms"] - bench["pieces"]["Cn|0"]["ms"]
        gm = json.loads((ROOT / f"results/drafting/stage4/global_multi-{prec}-p8.json").read_text(encoding="utf-8"))
        self.B = {0: 0.0, **{p: gm[f"P={p}"]["ms"] for p in range(1, 9)}}
        stage0 = json.loads((ROOT / "results/drafting/stage0/stage0.json").read_text(encoding="utf-8"))
        self.host_comb = stage0["host_cost_us"]["combined|8"]["median_us"]


def predict(t: dict, I: Inputs, lm_ms: float = 0.0, lm_host_us: float = 0.0) -> dict:
    """ms/byte of a tally (summed over prompts). `lm_ms`: drafter graph time per LM call."""
    loc = 0.0
    for key, n in t["by_kp"].items():
        kk, p, _, _ = (int(x) for x in key.split(","))
        loc += n * (I.pair[kk] + I.B[p] + (I.dp if p else 0.0))
    host = t["lookup_raw_calls"] * I.host_comb + t["lookup_flagged_calls"] * (I.host_comb + RETOK_US) + t["lm_calls"] * lm_host_us
    total = loc + t["rounds"] * I.o_round + host / 1000 + t["lm_calls"] * lm_ms
    E = t["emitted"]
    cross = t["kinds"].get("cross", 0)
    return {"ms_per_byte": I.prefill_pb + total / E, "bytes_per_round": E / t["rounds"], "bytes_per_global": E / t["global"] if t["global"] else None,
            "cross_share": cross / t["rounds"], "lm_calls_per_byte": t["lm_calls"] / E, "rounds_per_byte": t["rounds"] / E,
            "lm_per_round": t["lm_calls"] / t["rounds"], "emitted": E}


def lm_cost_rows(t: dict, cost: dict) -> dict:
    m = pct(t["m_hist"], 0.9) if t.get("m_hist") else 0
    if not m:
        return {c: 0.0 for c in COSTS} | {"m": 0}
    return {"measured": draft_cost(cost, m), "floor": m * cost["params"] * 2 / 448e9 * 1000, "free": 0.0, "m": m}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dol = Dolma2()
    parallel = json.loads((ROOT / "results/parallel/costmodel.json").read_text(encoding="utf-8"))
    old = json.loads((ROOT / "results/crossing/b1/costmodel.json").read_text(encoding="utf-8"))
    costs = {n: json.loads((B0 / f"cost_{n}.json").read_text(encoding="utf-8")) for n in NAMES}
    out: dict = {"bf16": {}, "int4": {}}
    md = ["# Patch-aligned rounds: cost model", ""]
    tallies: dict = {}
    for prec in ("bf16", "int4"):
        tallies[prec] = {}
        for nm in ("lookup", *NAMES):
            f = OUT / f"tallies_{prec}_{nm}.json"
            if f.exists():
                tallies[prec][nm] = json.loads(f.read_text(encoding="utf-8"))

    # ---- OLMo 2 (bf16): token rule with raw / healed / healed2 contexts (extended set)
    cases = load_truth(ROOT / "results/drafting/truth/bf16.json")
    olmo_truth = json.loads((ROOT / "results/drafting/truth/olmo-bf16.json").read_text(encoding="utf-8"))["prompts"]
    bt = statistics.mean(sum(len(dol.id_bytes(t)) for t in o["tokens"]["72"][1:73]) / 72 for o in olmo_truth.values() if o["set"] == "default")
    v0, v8 = olmo_verify(0), olmo_verify(8)
    pf_o = 2.681 - v0 / bt
    olmo: dict = {}
    for n in NAMES:
        rec = pickle.load(open(B0 / "drafts" / f"{n}.pkl", "rb"))["olmo"]
        h2 = pickle.load(open(ROOT / f"results/diagnostics/drafts/{n}-healed2.pkl", "rb"))["olmo"]
        for cm in ("raw", "healed", "healed2"):
            for k in KS_EVAL:
                eb = rounds = 0
                mh: dict = {}
                for pid, o in rec.items():
                    if cases[pid].set == "default":
                        continue
                    toks = o["tokens"]
                    if cm == "healed2":
                        cum = [0, *accumulate(len(dol.id_bytes(t)) for t in toks[1:])]
                        items = [h2[pid][e] for e in cum]
                    else:
                        items = o[cm]
                    d = PrecomputedTokenDrafter(items, dol)
                    d.set_text(o["prompt_bytes"], dol.id_bytes(toks[0]))
                    s = simulate_tokens(toks, d, K_TOK[k], ctx=[], max_new=len(toks) - 1)
                    eb += sum(len(dol.id_bytes(t)) for t in toks[1:1 + s.emitted])
                    rounds += s.rounds
                    for m, v in d.m_hist.items():
                        mh[m] = mh.get(m, 0) + v
                m = pct(mh, 0.9) if mh else 1
                bpr = eb / rounds
                rpb = 1 / bpr
                fl = m * costs[n]["params"] * 2 / 448e9 * 1000
                dc = draft_cost(costs[n], m)
                olmo[f"{n}|{cm}|{k}"] = {"bytes_per_round": bpr, "m": m,
                                         "measured": pf_o + rpb * (olmo_verify(K_TOK[k]) + dc + 0.11),
                                         "floor": pf_o + rpb * (olmo_verify(K_TOK[k]) + fl + 0.11),
                                         "free": pf_o + rpb * (olmo_verify(K_TOK[k]) + 0.11)}
    out["olmo"] = olmo
    olmo_best = {}
    for n in NAMES:
        for c in COSTS:
            olmo_best[f"{n}|{c}"] = min(((k, v) for k, v in olmo.items() if k.startswith(n + "|")), key=lambda kv: kv[1][c])
    olmo_lookup = 1.420  # combined K = 8, measured (drafting task)

    for prec in ("bf16", "int4"):
        I = Inputs(prec)
        T = tallies[prec]
        if "lookup" not in T:
            continue
        res: dict = {}
        # lookup-only policies
        for key, t in T["lookup"]["cells"].items():
            res[f"lookup|{key}"] = predict(t, I)
        # LM policies
        for n in NAMES:
            if n not in T:
                continue
            lmh = T[n].get("lm_host_us") or 0.0
            for key, t in T[n]["cells"].items():
                rows = lm_cost_rows(t, costs[n])
                for c in COSTS:
                    p = predict(t, I, rows[c], lmh)
                    res.setdefault(f"{n}|{key}", {})[c] = p
                res[f"{n}|{key}"]["m"] = rows["m"]
                res[f"{n}|{key}"]["lm_host_us"] = lmh
        out[prec] = res
        # sanity against Task A (P0 with the Task A combined drafter)
        chk = {}
        for s in ("default", "ext"):
            for k in KS_EVAL:
                mine = res[f"lookup|{s}|-|cut|{k}|P0"]["ms_per_byte"]
                ref = parallel[prec]["fused"][f"{s}|{k}"]["ms_per_byte"]
                chk[f"{s}|{k}"] = (mine, ref)
        out[prec + "_check_p0"] = chk
        md += [f"## {prec}", "", f"Prefill {I.prefill_pb:.3f} ms/byte, outside replays {I.o_round * 1000:.0f} us/round, lookup host {I.host_comb:.1f} us (+{RETOK_US:.0f} us with flags).", "",
               "P0 here vs Task A cost model (fused verify), ms/byte: " + "; ".join(f"{s}: {m_:.3f} vs {r_:.3f}" for s, (m_, r_) in chk.items()), ""]

        # ---- reproduction of the old L1 cells (healed, full) as a check of this port
        if "smollm2-135m" in T:
            md += ["Port check, L1 (= old `multi`) vs `results/crossing/b1/costmodel.json` (ms/byte, ext; measured / floor / free):", ""]
            for n in NAMES:
                if n not in T:
                    continue
                for cm in ("raw", "healed"):
                    k = 32
                    mine = res[f"{n}|ext|{cm}|full|{k}|L1"]
                    o = old["bolmo"][prec]["cells"][f"{n}|{cm}|full|{k}|multi|ext"]
                    md.append(f"- {n} {cm} full K={k}: new {mine['measured']['ms_per_byte']:.3f} / {mine['floor']['ms_per_byte']:.3f} / {mine['free']['ms_per_byte']:.3f}; "
                              f"old {o['ms_per_byte']:.3f} / {o['ms_floor']:.3f} / {o['ms_free']:.3f}; bytes/round {mine['measured']['bytes_per_round']:.2f} vs {o['bytes_per_round']:.2f}")
            md.append("")

        # ---- best P0 on ext / default
        p0 = {s: min(((k, res[f"lookup|{s}|-|cut|{k}|P0"]) for k in KS_EVAL), key=lambda kv: kv[1]["ms_per_byte"]) for s in ("ext", "default")}
        md += [f"Best P0 (combined lookup, within-patch): ext K={p0['ext'][0]} {p0['ext'][1]['ms_per_byte']:.3f} ms/byte ({p0['ext'][1]['bytes_per_round']:.2f} B/round); "
               f"default K={p0['default'][0]} {p0['default'][1]['ms_per_byte']:.3f}.", ""]
        out[prec + "_p0"] = {s: {"k": k, **v} for s, (k, v) in p0.items()}

        # ---- H43: best L1 cell per cost row, contexts old vs new
        md += ["### Token-LM crossing (L1), best cell per cost row (chosen on ext; flags cut or full as in Task B)", "",
               "| precision | drafter cost | contexts | cell | B/round ext | ms/byte ext | vs P0 ext | ms/byte default | vs P0 default |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
        h43 = {}
        for c in COSTS:
            for label, cms in (("raw, healed (Task B)", ("raw", "healed")), ("raw, healed, healed2", ("raw", "healed", "healed2"))):
                cand = [(n, cm, fm, k) for n in NAMES if n in T for cm in cms for fm in ("cut", "full") for k in KS_EVAL
                        if f"{n}|ext|{cm}|{fm}|{k}|L1" in res]
                n, cm, fm, k = min(cand, key=lambda x: res[f"{x[0]}|ext|{x[1]}|{x[2]}|{x[3]}|L1"][c]["ms_per_byte"])
                r = res[f"{n}|ext|{cm}|{fm}|{k}|L1"][c]
                rd = res.get(f"{n}|default|{cm}|{fm}|{k}|L1", {}).get(c)
                h43[f"{c}|{label}"] = {"cell": f"{n}|{cm}|{fm}|{k}", "ext": r["ms_per_byte"], "bpr": r["bytes_per_round"],
                                       "ratio_p0_ext": r["ms_per_byte"] / p0["ext"][1]["ms_per_byte"],
                                       "default": rd["ms_per_byte"] if rd else None,
                                       "ratio_p0_default": rd["ms_per_byte"] / p0["default"][1]["ms_per_byte"] if rd else None}
                x = h43[f"{c}|{label}"]
                md.append(f"| {prec} | {c} | {label} | {x['cell']} | {x['bpr']:.2f} | {x['ext']:.3f} | {x['ratio_p0_ext']:.2f}x | "
                          + (f"{x['default']:.3f} | {x['ratio_p0_default']:.2f}x |" if x["default"] else "n/a (no healed2 recording) | |"))
        out[prec + "_h43"] = h43
        md.append("")

        # ---- byte tax: Bolmo L1 vs OLMo, same LM drafter (bf16 only: OLMo numbers exist for bf16)
        if prec == "bf16":
            md += ["### Byte tax with the same LM drafter (ext prompts; Bolmo L1, OLMo token rule; best cell of each over K and contexts)", "",
                   "| drafter cost | drafter | contexts | Bolmo ms/byte | OLMo ms/byte | ratio |", "|---|---|---|---:|---:|---:|"]
            tax = {}
            for c in COSTS:
                for label, cms in (("raw, healed", ("raw", "healed")), ("raw, healed, healed2", ("raw", "healed", "healed2"))):
                    best = None
                    for n in NAMES:
                        if n not in T:
                            continue
                        bb = min(res[f"{n}|ext|{cm}|{fm}|{k}|L1"][c]["ms_per_byte"] for cm in cms for fm in ("cut", "full") for k in KS_EVAL)
                        oo = min(v[c] for kk, v in olmo.items() if kk.startswith(n + "|") and kk.split("|")[1] in cms)
                        if best is None or bb / oo < best[1] / best[2]:
                            best = (n, bb, oo)
                    tax[f"{c}|{label}"] = {"drafter": best[0], "bolmo": best[1], "olmo": best[2], "ratio": best[1] / best[2]}
                    md.append(f"| {c} | {best[0]} | {label} | {best[1]:.3f} | {best[2]:.3f} | {best[1] / best[2]:.2f} |")
            # all drafters, healed2, measured
            md.append("")
            for n in NAMES:
                if n in T:
                    bb = min(res[f"{n}|ext|healed2|{fm}|{k}|L1"]["measured"]["ms_per_byte"] for fm in ("cut", "full") for k in KS_EVAL)
                    oo = min(v["measured"] for kk, v in olmo.items() if kk.startswith(n + "|") and kk.split("|")[1] == "healed2")
                    tax[f"each|{n}"] = {"bolmo": bb, "olmo": oo, "ratio": bb / oo}
                    md.append(f"- {n}, healed2, measured: Bolmo {bb:.3f}, OLMo {oo:.3f}, ratio {bb / oo:.2f}")
            out["byte_tax"] = tax
            md.append("")

        # ---- policy table
        md += [f"### Policy table, {prec}, extended set (default prompts where the recording exists); flags `cut` unless noted", ""]
        pol_rows = []

        def row(label, key_ext, costkey=None):
            e = res[key_ext]
            e = e[costkey] if costkey else e
            d_key = key_ext.replace("|ext|", "|default|") if "|ext|" in key_ext else key_ext.replace("ext|", "default|", 1)
            d = res.get(d_key)
            d = (d[costkey] if costkey else d) if d else None
            return {"label": label, "key": key_ext, "ext": e, "default": d}

        for k in SHOW_K:
            pol_rows.append(row(f"P0 K={k}", f"lookup|ext|-|cut|{k}|P0"))
            pol_rows.append(row(f"P1 K={k}", f"lookup|ext|-|cut|{k}|P1"))
            pol_rows.append(row(f"P4 K={k}", f"lookup|ext|-|cut|{k}|P4"))
            pol_rows.append(row(f"P4* (oracle lookup) K={k}", f"lookup|ext|-|cut|{k}|P4*"))
        md += ["| policy | B/round | B/global pass | crossing rounds | ms/byte ext | vs P0 | ms/byte default | vs P0 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in pol_rows:
            e, d = r["ext"], r["default"]
            md.append(f"| {r['label']} | {e['bytes_per_round']:.2f} | {e['bytes_per_global']:.1f} | {e['cross_share'] * 100:.0f}% | {e['ms_per_byte']:.3f} | "
                      f"{(1 - e['ms_per_byte'] / p0['ext'][1]['ms_per_byte']) * 100:+.1f}% | "
                      + (f"{d['ms_per_byte']:.3f} | {(1 - d['ms_per_byte'] / p0['default'][1]['ms_per_byte']) * 100:+.1f}% |" if d else "n/a | |"))
        md.append("")
        out[prec + "_policy_rows"] = [{"label": r["label"], "ext": r["ext"], "default": r["default"]} for r in pol_rows]

        # lookup policies with K chosen on ext
        md += ["Lookup policies, K chosen on the extended set (default prompts = the speed workload with that K); "
               "ratio = Bolmo default-prompt ms/byte over OLMo 2 + combined lookup K = 8 (1.420 measured; bf16 only):", "",
               "| policy | K | B/round (ext) | B/global pass | crossing rounds | ms/byte ext | vs P0 ext | ms/byte default | vs P0 default | Bolmo/OLMo |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        sel_look = {}
        for pol in ("P0", "P1", "P4", "P4*"):
            k = min(KS_EVAL, key=lambda kk: res[f"lookup|ext|-|cut|{kk}|{pol}"]["ms_per_byte"])
            e, d = res[f"lookup|ext|-|cut|{k}|{pol}"], res[f"lookup|default|-|cut|{k}|{pol}"]
            sel_look[pol] = {"k": k, "ext": e, "default": d}
        for pol, v in sel_look.items():
            e, d, k = v["ext"], v["default"], v["k"]
            ratio = f"{d['ms_per_byte'] / olmo_lookup:.2f}" if prec == "bf16" else "n/a"
            md.append(f"| {pol}{' (oracle lookup, upper bound)' if pol.endswith('*') else ''} | {k} | {e['bytes_per_round']:.2f} | {e['bytes_per_global']:.1f} | {e['cross_share'] * 100:.0f}% | "
                      f"{e['ms_per_byte']:.3f} | {(1 - e['ms_per_byte'] / sel_look['P0']['ext']['ms_per_byte']) * 100:+.1f}% | {d['ms_per_byte']:.3f} | "
                      f"{(1 - d['ms_per_byte'] / sel_look['P0']['default']['ms_per_byte']) * 100:+.1f}% | {ratio} |")
        md.append("")
        out[prec + "_sel_lookup"] = sel_look

        # LM policies: per drafter, per cost row, best cell by ms/byte on ext (contexts incl. healed2), flags cut
        md += ["LM policies, best cell per (policy, drafter, cost row) on ext (flags `cut`; K and context chosen on ext):", "",
               "| policy | drafter | cost | cell | B/round | B/global pass | crossing | LM calls/byte | ms/byte ext | vs P0 ext | ms/byte default | vs P0 default | Bolmo/OLMo (same drafter, ext) |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        best_lm = {}
        for pol in ("L1", "P2", "P2b", "P3", "P2*", "P3*"):
            for n in NAMES:
                if n not in T:
                    continue
                for c in COSTS:
                    cand = [(cm, k) for cm in ("raw", "healed", "healed2") for k in KS_EVAL if f"{n}|ext|{cm}|cut|{k}|{pol}" in res]
                    if not cand:
                        continue
                    cm, k = min(cand, key=lambda x: res[f"{n}|ext|{x[0]}|cut|{x[1]}|{pol}"][c]["ms_per_byte"])
                    e = res[f"{n}|ext|{cm}|cut|{k}|{pol}"][c]
                    d = res.get(f"{n}|default|{cm}|cut|{k}|{pol}")
                    d = d[c] if d else None
                    ob = min(v[c] for kk, v in olmo.items() if kk.startswith(n + "|")) if prec == "bf16" else None
                    best_lm[f"{pol}|{n}|{c}"] = {"cell": f"{cm}|cut|{k}", "ext": e, "default": d, "olmo_ms": ob}
                    md.append(f"| {pol} | {n} | {c} | {cm}, K={k} | {e['bytes_per_round']:.2f} | {e['bytes_per_global']:.1f} | {e['cross_share'] * 100:.0f}% | {e['lm_calls_per_byte']:.3f} | "
                              f"{e['ms_per_byte']:.3f} | {(1 - e['ms_per_byte'] / p0['ext'][1]['ms_per_byte']) * 100:+.1f}% | "
                              + (f"{d['ms_per_byte']:.3f} | {(1 - d['ms_per_byte'] / p0['default'][1]['ms_per_byte']) * 100:+.1f}% |" if d else "n/a | |")
                              + (f" {e['ms_per_byte'] / ob:.2f} |" if ob else " n/a |"))
        out[prec + "_best_lm"] = best_lm
        md.append("")

        # acceptance table (H44): bytes per round for every LM policy, drafter, ctx, flags at K=32
        if prec == "bf16":
            md += ["### Acceptance, bytes per round (ext, K = 32, bf16)", "", "| policy | drafter | ctx | cut | full |", "|---|---|---|---:|---:|"]
            for pol in ("L1", "P2", "P2b", "P3", "P2*", "P3*"):
                for n in NAMES:
                    if n not in T:
                        continue
                    for cm in ("raw", "healed", "healed2"):
                        a_ = res.get(f"{n}|ext|{cm}|cut|32|{pol}")
                        b_ = res.get(f"{n}|ext|{cm}|full|32|{pol}")
                        if a_ and b_:
                            md.append(f"| {pol} | {n} | {cm} | {a_['measured']['bytes_per_round']:.2f} | {b_['measured']['bytes_per_round']:.2f} |")
            md.append("")

    # OLMo lookup baseline for the lookup policies' ratio
    out["olmo_lookup_k8"] = olmo_lookup
    (OUT / "costmodel.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    (OUT / "costmodel.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
