"""Stage 0 step 4 (TASK_build_p4.md): the patch-aligned cost model rerun with the measured crossing-round cost.

round cost = Task A's verify pair(K) + B(P) + (C_p - C_n)  [as tools/aligned_costmodel.py]
           + extra(K, P) for a crossing round with P >= 2 patches      <- Stage 0 (results/p4/stage0/*.json)
           + outside replays + lookup host cost (+ the re-tokenization for the flags).

Tallies: results/aligned/tallies_<prec>_lookup.json (P0, and P4 with the same K for both round types, as predicted by the
patch-aligned task) and results/p4/tallies_<prec>.json (P4 with k_max within the patch and k_cross across patches).
Output: results/p4/costmodel.{json,md}.   Usage: python tools/p4_costmodel.py [--retok-us 150] [--out results/p4]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import aligned_costmodel as acm  # noqa: E402

KS_EVAL = acm.KS_EVAL
OLMO_LOOKUP = 1.420  # OLMo 2 + combined lookup K = 8, bf16 (drafting task)


def load_extra(prec: str, clamp: bool = True) -> dict:
    """(K, P) -> measured extra ms of a crossing round (same-session: X - (pair + B(P) + dp))."""
    ex: dict = {}
    for f in sorted((ROOT / "results/p4/stage0").glob(f"{prec}-k*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        for key, c in r["cells"].items():
            k, p = (int(x) for x in key.split("|"))
            ex[(k, p)] = max(c["extra_now"], 0.0) if clamp else c["extra_now"]
    return ex


def extra_for(ex: dict, kk: int, p: int) -> float:
    if p < 2 or not ex:
        return 0.0
    ks = sorted({k for k, _ in ex})
    def at(k):
        q = min(p, max(pp for kk_, pp in ex if kk_ == k))
        return ex.get((k, q), 0.0)
    if kk <= ks[0]:
        return at(ks[0])
    if kk >= ks[-1]:
        return at(ks[-1])
    for a, b in zip(ks, ks[1:]):
        if a <= kk <= b:
            return at(a) + (at(b) - at(a)) * (kk - a) / (b - a)
    return 0.0


def predict(t: dict, I: acm.Inputs, ex: dict, retok_us: float) -> dict:
    loc = 0.0
    extra_total = 0.0
    for key, n in t["by_kp"].items():
        kk, p, cross, _ = (int(x) for x in key.split(","))
        e = extra_for(ex, kk, p) if cross else 0.0
        loc += n * (I.pair[kk] + I.B[p] + (I.dp if p else 0.0) + e)
        extra_total += n * e
    host = t["lookup_raw_calls"] * I.host_comb + t["lookup_flagged_calls"] * (I.host_comb + retok_us)
    total = loc + t["rounds"] * I.o_round + host / 1000
    E = t["emitted"]
    return {"ms_per_byte": I.prefill_pb + total / E, "bytes_per_round": E / t["rounds"], "bytes_per_global": E / t["global"] if t["global"] else None,
            "cross_share": t["kinds"].get("cross", 0) / t["rounds"], "emitted": E, "extra_ms_per_byte": extra_total / E,
            "rounds_per_byte": t["rounds"] / E}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retok-us", type=float, default=acm.RETOK_US)
    ap.add_argument("--out", default="results/p4")
    a = ap.parse_args()
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"retok_us": a.retok_us}
    md = ["# P4: updated cost model (Stage 0)", "", f"Re-tokenization host time per flagged lookup round: {a.retok_us:.0f} us.", ""]
    for prec in ("bf16", "int4"):
        I = acm.Inputs(prec)
        ex = load_extra(prec)
        ex_raw = load_extra(prec, clamp=False)
        al = json.loads((ROOT / f"results/aligned/tallies_{prec}_lookup.json").read_text(encoding="utf-8"))["cells"]
        p4 = json.loads((ROOT / f"results/p4/tallies_{prec}.json").read_text(encoding="utf-8"))["cells"]
        r: dict = {}
        md += [f"## {prec}", ""]
        if ex_raw:
            md.append("Measured extra of a crossing round (ms, same session, X - (pair + B(P) + dp); negative values are clamped to 0 in the model): "
                      + "; ".join(f"K={k} " + ", ".join(f"P{p}: {ex_raw[(k, p)]:+.2f}" for p in range(2, 9) if (k, p) in ex_raw) for k in sorted({k for k, _ in ex_raw})))
            md.append("")
        # P0
        p0 = {}
        for s in ("ext", "default"):
            for k in KS_EVAL:
                p0[(s, k)] = predict(al[f"{s}|-|cut|{k}|P0"], I, ex, a.retok_us)
        k0 = min(KS_EVAL, key=lambda k: p0[("ext", k)]["ms_per_byte"])
        r["P0"] = {"k": k0, "ext": p0[("ext", k0)], "default": p0[("default", k0)], "all": {f"{s}|{k}": v for (s, k), v in p0.items()}}
        # P4, aligned-task version (same K for both round types) with and without the measured extra
        for label, exx in (("P4 same K, no extra (patch-aligned prediction)", {}), ("P4 same K, + measured extra", ex)):
            cells = {(s, k): predict(al[f"{s}|-|cut|{k}|P4"], I, exx, a.retok_us) for s in ("ext", "default") for k in KS_EVAL}
            kb = min(KS_EVAL, key=lambda k: cells[("ext", k)]["ms_per_byte"])
            r[label] = {"k": kb, "ext": cells[("ext", kb)], "default": cells[("default", kb)],
                        "all": {f"{s}|{k}": v for (s, k), v in cells.items()}}
        # P4 as built: k_max within, k_cross across
        built = {}
        for key, t in p4.items():
            s, _, kc, kw = key.split("|")
            built[(s, int(kc), kw)] = predict(t, I, ex, a.retok_us)
        kcs = sorted({kc for _, kc, _ in built})
        kws = sorted({kw for _, _, kw in built})
        best = min(((kc, kw) for kc in kcs for kw in kws if ("ext", kc, kw) in built), key=lambda x: built[("ext", x[0], x[1])]["ms_per_byte"])
        r["P4 as built"] = {"k_cross": best[0], "k_within": best[1], "ext": built[("ext", *best)], "default": built[("default", *best)],
                            "all": {f"{s}|{kc}|{kw}": v for (s, kc, kw), v in built.items()}}
        res[prec] = r
        md += ["| policy | K (cross / within) | B/round ext | B/global ext | crossing rounds | extra ms/byte | ms/byte ext | vs P0 ext | ms/byte default | vs P0 default | Bolmo/OLMo (default) |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        p0e, p0d = r["P0"]["ext"]["ms_per_byte"], r["P0"]["default"]["ms_per_byte"]
        rows = [("P0 within-patch", f"{r['P0']['k']}", r["P0"])]
        for label in ("P4 same K, no extra (patch-aligned prediction)", "P4 same K, + measured extra"):
            rows.append((label, f"{r[label]['k']} / {r[label]['k']}", r[label]))
        rows.append(("P4 as built", f"{r['P4 as built']['k_cross']} / {r['P4 as built']['k_within']}", r["P4 as built"]))
        for label, kstr, v in rows:
            e, d = v["ext"], v["default"]
            ratio = f"{d['ms_per_byte'] / OLMO_LOOKUP:.2f}" if prec == "bf16" else "n/a"
            md.append(f"| {label} | {kstr} | {e['bytes_per_round']:.2f} | {e['bytes_per_global']:.1f} | {e['cross_share'] * 100:.0f}% | {e['extra_ms_per_byte']:.3f} | "
                      f"{e['ms_per_byte']:.3f} | {(1 - e['ms_per_byte'] / p0e) * 100:+.1f}% | {d['ms_per_byte']:.3f} | {(1 - d['ms_per_byte'] / p0d) * 100:+.1f}% | {ratio} |")
        md.append("")
        # the grid
        md += [f"P4 as built, ms/byte (ext / default) by k_cross (rows) and k_within (columns; `-` = same as k_cross):", "",
               "| k_cross | " + " | ".join(kws) + " |", "|---|" + "---:|" * len(kws)]
        for kc in kcs:
            md.append(f"| {kc} | " + " | ".join((f"{built[('ext', kc, kw)]['ms_per_byte']:.3f} / {built[('default', kc, kw)]['ms_per_byte']:.3f}" if ("ext", kc, kw) in built else "") for kw in kws) + " |")
        md.append("")
    # gate
    g = res["bf16"]["P4 as built"]["default"]["ms_per_byte"]
    g0 = res["bf16"]["P0"]["default"]["ms_per_byte"]
    gate = (1 - g / g0) * 100
    res["gate"] = {"bf16_default_gain_pct": gate, "passes": gate >= 15.0, "p4_ms": g, "p0_ms": g0}
    md += [f"**Gate (bf16, default prompts, P4 as built vs P0): {gate:+.1f}% (needs >= +15%) -> {'PASS' if gate >= 15 else 'STOP'}**", ""]
    (out_dir / "costmodel.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (out_dir / "costmodel.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
