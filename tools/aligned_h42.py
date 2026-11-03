"""Patch-aligned task, step 1 (H42): where do Bolmo's rounds restart, and what do they emit (healed2 contexts)?

Usage: python tools/aligned_h42.py [--names smollm2-360m,qwen2.5-0.5b,smollm2-135m] [--mode healed2]
Output: results/aligned/h42_<mode>.json and h42_<mode>.md (CPU only; replays recorded continuations).

Rounds are those of `diagnostics.replay_byte_rounds` (byte rule, crossing, K = 32, at most 8 patches). A round restarts
*at a token end* when its last emitted byte ends a dolma2 token of the retokenized text, otherwise *mid-token* with
1, 2, 3 or 4+ bytes of that token still to come. Counterfactuals, one round from each recorded mid-token start:

* actual: 1 + L
* unit: the free unit completes the token (the round also emits the rest of the token that its last byte is in),
  everything else as recorded: the effect of the one-byte free unit and of the remaining-token length
* oracle-in-token: the draft is the true rest of the patch (an oracle lookup) followed by the LM's continuation
  recorded at that boundary: the effect of drafter accuracy inside the remaining token (also gains those bytes)
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from byteprof.drafting import byte_of, flag_of  # noqa: E402
from byteprof.drafting.aligned import RecordedSources, plan_round  # noqa: E402
from byteprof.drafting.diagnostics import BYTE_ENDS, replay_byte_rounds, retokenize  # noqa: E402
from byteprof.drafting.flags import Dolma2  # noqa: E402
from byteprof.drafting.lmdraft import PrecomputedLMDrafter  # noqa: E402
from byteprof.drafting.sim import accepted_prefix  # noqa: E402
from byteprof.drafting.truthio import load_truth  # noqa: E402

K, MAX_PATCHES, N = 32, 8, 256
OUT = ROOT / "results/aligned"
DRAFTS = ROOT / "results/crossing/b0/drafts"
DIAG = ROOT / "results/diagnostics/drafts"
CLASSES = ("token_end", "rem1", "rem2", "rem3", "rem4+")


def classify(spans, pos: int) -> str | None:
    """Class of the restart whose last emitted byte is at data position `pos`."""
    for s, e in spans:
        if s <= pos < e:
            if pos == e - 1:
                return "token_end"
            r = e - 1 - pos
            return f"rem{r}" if r < 4 else "rem4+"
    return None


def remaining(spans, pos: int) -> int:
    for s, e in spans:
        if s <= pos < e:
            return e - 1 - pos
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="smollm2-135m,smollm2-360m,qwen2.5-0.5b")
    ap.add_argument("--mode", default="healed2")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dol = Dolma2()
    cases = load_truth(ROOT / "results/drafting/truth/bf16.json")
    ids = [p for p, c in cases.items() if c.set != "default"]
    res: dict = {}
    for name in [x for x in a.names.split(",") if x]:
        if a.mode == "healed2":
            items = pickle.load(open(DIAG / f"{name}-healed2.pkl", "rb"))["bolmo"]
        else:
            items = {p: v[a.mode] for p, v in pickle.load(open(DRAFTS / f"{name}.pkl", "rb"))["bolmo"].items()}
        res[name] = {}
        for fm in ("cut", "full"):
            agg: dict = {c: {"rounds": 0, "bytes": 0, "L": 0, "L0": 0, "ends": defaultdict(int), "unit": 0, "orc": 0,
                             "rem_sum": 0, "first_hit": 0} for c in CLASSES}
            for pid in ids:
                c = cases[pid]
                truth = c.truth[N]
                P = len(c.text.encode("utf-8"))
                data = c.text.encode("utf-8") + bytes(byte_of(x) for x in truth[1:])
                _, spans, _ = retokenize(dol, data)
                recs = replay_byte_rounds(truth, PrecomputedLMDrafter(items[pid], dol, fm), K, ctx_atoms=c.ctx_atoms,
                                          multi_patch=True, max_patches=MAX_PATCHES)
                src = RecordedSources(_Dummy(), dol, c.ctx_atoms, truth, items[pid], fm)
                for r in recs:
                    pos = P - 1 + r.i
                    cl = classify(spans, pos)
                    if cl is None:
                        continue
                    g = agg[cl]
                    g["rounds"] += 1
                    g["bytes"] += r.emitted
                    g["L"] += r.L
                    g["L0"] += r.L == 0
                    g["first_hit"] += r.L >= 1
                    g["ends"][r.end] += 1
                    if cl != "token_end":
                        rem = remaining(spans, pos)
                        g["rem_sum"] += rem
                        # unit: the free unit completes the token that the last emitted byte is in
                        last = pos + r.emitted
                        tok_end_after = next((e for s, e in spans if s <= last < e), last + 1) - 1  # data pos of token's last byte
                        g["unit"] += max(r.emitted, min(tok_end_after - pos, len(truth) - 1 - r.i))
                        # oracle-in-token
                        plan = plan_round("P2", r.i, truth, src, K, oracle=True)
                        draft = plan.draft
                        from byteprof.drafting.diagnostics import limit_patches

                        draft, _ = limit_patches(draft, flag_of(truth[r.i]), MAX_PATCHES)
                        L = accepted_prefix(draft, truth[r.i + 1: r.i + 2 + K], K, True)
                        g["orc"] += min(1 + L, len(truth) - 1 - r.i)
            res[name][fm] = agg
    out = {}
    md = [f"# H42: rounds by restart position ({a.mode} contexts, extended set, K = {K}, crossing rule)", ""]
    for name, byfm in res.items():
        out[name] = {}
        for fm, agg in byfm.items():
            rows = {}
            for cl in CLASSES:
                g = agg[cl]
                n = max(g["rounds"], 1)
                rows[cl] = {"rounds": g["rounds"], "bytes_per_round": g["bytes"] / n, "mean_L": g["L"] / n,
                            "first_byte_hit": g["first_hit"] / n, "ends": {k: v / n for k, v in g["ends"].items()},
                            "unit_bytes": g["unit"] / n if cl != "token_end" else None,
                            "oracle_bytes": g["orc"] / n if cl != "token_end" else None,
                            "mean_remaining": g["rem_sum"] / n if cl != "token_end" else None}
            mid = [agg[c] for c in CLASSES[1:]]
            nm = max(sum(g["rounds"] for g in mid), 1)
            rows["mid_all"] = {"rounds": sum(g["rounds"] for g in mid), "bytes_per_round": sum(g["bytes"] for g in mid) / nm,
                               "mean_L": sum(g["L"] for g in mid) / nm, "first_byte_hit": sum(g["first_hit"] for g in mid) / nm,
                               "unit_bytes": sum(g["unit"] for g in mid) / nm, "oracle_bytes": sum(g["orc"] for g in mid) / nm,
                               "mean_remaining": sum(g["rem_sum"] for g in mid) / nm,
                               "ends": {k: sum(g["ends"].get(k, 0) for g in mid) / nm for k in BYTE_ENDS}}
            tot_rounds = rows["token_end"]["rounds"] + rows["mid_all"]["rounds"]
            rows["all"] = {"rounds": tot_rounds,
                           "bytes_per_round": (agg["token_end"]["bytes"] + sum(g["bytes"] for g in mid)) / max(tot_rounds, 1),
                           "mid_share": rows["mid_all"]["rounds"] / max(tot_rounds, 1)}
            out[name][fm] = rows
            b, m = rows["token_end"], rows["mid_all"]
            shortfall = b["bytes_per_round"] - m["bytes_per_round"]
            u = m["unit_bytes"] - m["bytes_per_round"]
            rows["decomposition"] = {"shortfall": shortfall, "ratio_mid_over_boundary": m["bytes_per_round"] / b["bytes_per_round"],
                                     "unit_gain": u, "unit_share_of_shortfall": u / shortfall if shortfall else None,
                                     "residual_accuracy": shortfall - u,
                                     "oracle_gain": m["oracle_bytes"] - m["bytes_per_round"]}
            md += [f"## {name} / flags `{fm}`", "", "| restart | rounds | B/round | mean L | first byte right | wrong_byte | wrong_flag | cap | exhausted | patch_cap | unit B | oracle-in-token B |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for cl in (*CLASSES, "mid_all"):
                r = rows[cl]
                e = r["ends"]
                md.append(f"| {cl} | {r['rounds']} | {r['bytes_per_round']:.2f} | {r['mean_L']:.2f} | {r['first_byte_hit'] * 100:.0f}% | "
                          f"{e.get('wrong_byte', 0) * 100:.0f}% | {e.get('wrong_flag', 0) * 100:.0f}% | {e.get('cap', 0) * 100:.0f}% | "
                          f"{e.get('exhausted', 0) * 100:.0f}% | {e.get('patch_cap', 0) * 100:.0f}% | "
                          + (f"{r['unit_bytes']:.2f} | {r['oracle_bytes']:.2f} |" if r.get("unit_bytes") is not None else "- | - |"))
            d = rows["decomposition"]
            md += ["", f"All rounds {rows['all']['bytes_per_round']:.2f} B/round, {rows['all']['mid_share'] * 100:.0f}% restart mid-token. "
                   f"Mid-token / token-end = {d['ratio_mid_over_boundary']:.2f}; shortfall {d['shortfall']:.2f} B; the unit counterfactual "
                   f"(free unit completes the token, mean {m['mean_remaining']:.2f} B remaining) closes {d['unit_gain']:.2f} B = "
                   f"{(d['unit_share_of_shortfall'] or 0) * 100:.0f}% of it; the rest, {d['residual_accuracy']:.2f} B, is what the drafter does not get right. "
                   f"Oracle-in-token adds {d['oracle_gain']:.2f} B.", ""]
    (OUT / f"h42_{a.mode}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (OUT / f"h42_{a.mode}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


class _Dummy:
    def reset(self, *a, **k):
        pass

    def push(self, *a, **k):
        pass

    def propose(self, k):
        return []


if __name__ == "__main__":
    sys.exit(main())
