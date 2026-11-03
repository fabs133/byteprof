"""Patch-aligned task, step 6 (H47): the two-state model of bytes per round.

Usage: python tools/aligned_twostate.py   -> results/aligned/twostate.{json,md}

States: the round restarts at a boundary (the last emitted atom ends a patch / a dolma2 token) or mid-patch. The model
(`aligned.two_state_bytes_per_round`) takes the transition counts between the start states of consecutive rounds and the
mean bytes per round in each state. Reported per cell:

* in-sample: parameters fitted on all extended-set prompts, bytes per round predicted for the same prompts
* held-out: parameters fitted on the even-numbered prompts, predicted for the odd ones and vice versa
  (bytes per round of the held-out half measured by the simulation)

Bolmo cells: the policies with the combined lookup drafter and with a recorded token-LM drafter (healed2, bf16, K = 32 for
the crossing rules, K = 16 for lookup crossing). OLMo 2: its text under Bolmo's byte rule (same recorded continuations,
byte-level restarts, every restart state defined by dolma2 token ends) and its own token rule, which has one state only
(every round restarts on a token end), so the model is trivially exact there.

The analytic figure for the paper (bytes per round against the hit rate at boundaries, free unit of 1 byte and of 1 token)
is tabulated from `aligned.two_state_figure` with the hit rates and restart shares measured here.
"""

import json
import pickle
import sys
from itertools import accumulate
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from byteprof.drafting import byte_of, make_atom  # noqa: E402
from byteprof.drafting.aligned import (RecordedSources, simulate_policy, two_state_bytes_per_round,  # noqa: E402
                                       two_state_figure)
from byteprof.drafting.diagnostics import atoms_with_flags, replay_byte_rounds  # noqa: E402
from byteprof.drafting.drafters import CombinedDrafter, ContextDrafter  # noqa: E402
from byteprof.drafting.flags import Dolma2  # noqa: E402
from byteprof.drafting.lmdraft import PrecomputedLMDrafter  # noqa: E402
from byteprof.drafting.ngram import NGramDrafter, NGramModel  # noqa: E402
from byteprof.drafting.truthio import load_truth  # noqa: E402

OUT = ROOT / "results/aligned"
KS_ALL = (1, 2, 4, 6, 8, 12, 16, 24, 32)
NAMES = ("smollm2-135m", "smollm2-360m", "qwen2.5-0.5b")


def fit_predict(fit_rows, test_rows):
    """Rows: (start_rounds, start_bytes, trans) dicts per prompt. Returns (predicted, actual) bytes per round of test_rows."""
    sr = {0: 0, 1: 0}
    sb = {0: 0, 1: 0}
    tr: dict = {}
    for r_, b_, t_ in fit_rows:
        for s in (0, 1):
            sr[s] += r_[s]
            sb[s] += b_[s]
        for k, v in t_.items():
            tr[k] = tr.get(k, 0) + v
    per = {s: (sb[s] / sr[s] if sr[s] else 0.0) for s in (0, 1)}
    pred = two_state_bytes_per_round({k: float(v) for k, v in tr.items()}, per)
    act = sum(sum(b_.values()) for _, b_, _ in test_rows) / max(sum(sum(r_.values()) for r_, _, _ in test_rows), 1)
    return pred, act, per, sr, tr


def evaluate(rows):
    ins_p, ins_a, per, sr, tr = fit_predict(rows, rows)
    even, odd = rows[0::2], rows[1::2]
    p1, a1, *_ = fit_predict(even, odd)
    p2, a2, *_ = fit_predict(odd, even)
    ho_p = (p1 * len(odd) + p2 * len(even)) / len(rows)  # unweighted mix; errors reported per fold as well
    errs = [abs(p1 / a1 - 1), abs(p2 / a2 - 1)]
    tot = sum(sr.values())
    return {"in_sample": {"pred": ins_p, "actual": ins_a, "rel_err": ins_p / ins_a - 1},
            "held_out": {"folds": [{"pred": p1, "actual": a1}, {"pred": p2, "actual": a2}], "max_rel_err": max(errs)},
            "e_state": {"mid": per[0], "boundary": per[1]}, "share_boundary_rounds": sr[1] / tot if tot else None,
            "trans": {f"{a}{b}": v for (a, b), v in tr.items()}}


def main() -> int:
    dol = Dolma2()
    cases = load_truth(ROOT / "results/drafting/truth/bf16.json")
    ext = [c for c in cases.values() if c.set != "default"]
    corpus = ROOT / "results/drafting/corpus"
    model = NGramModel.build((corpus / "corpus.bin").read_bytes(), np.fromfile(corpus / "flags.bin", np.uint8),
                             (1, 2, 3, 4, 5, 6, 8, 12), bits=24)

    def lookup():
        return CombinedDrafter(ContextDrafter(min_len=3), NGramDrafter(model, min_count=1, max_order=6))

    res: dict = {"bolmo": {}, "olmo": {}}
    md = ["# H47: two-state model of bytes per round", "",
          "| model / policy | drafter | in-sample pred / sim | rel. err | held-out max rel. err | e(mid) | e(boundary) | boundary share of rounds |", "|---|---|---:|---:|---:|---:|---:|---:|"]

    def add_row(model_, label, drafter, ev):
        i = ev["in_sample"]
        md.append(f"| {model_} {label} | {drafter} | {i['pred']:.3f} / {i['actual']:.3f} | {i['rel_err'] * 100:+.1f}% | {ev['held_out']['max_rel_err'] * 100:.1f}% | "
                  f"{ev['e_state']['mid']:.2f} | {ev['e_state']['boundary']:.2f} | {ev['share_boundary_rounds'] * 100:.0f}% |")

    # Bolmo, lookup policies
    for pol, k in (("P0", 8), ("P1", 16), ("P4", 16), ("P4", 32)):
        rows = []
        for c in ext:
            src = RecordedSources(lookup(), dol, c.ctx_atoms, c.truth[256], None, "cut")
            r = simulate_policy(c.truth[256], pol, src, k, max_new=256, ks=[x for x in KS_ALL if x <= k])
            rows.append((r.start_rounds, r.start_bytes, r.trans))
        ev = evaluate(rows)
        res["bolmo"][f"{pol}|lookup|K{k}"] = ev
        add_row("Bolmo", f"{pol} K={k}", "combined lookup", ev)
    # Bolmo, LM policies (healed2)
    for n in NAMES:
        h2 = pickle.load(open(ROOT / f"results/diagnostics/drafts/{n}-healed2.pkl", "rb"))["bolmo"]
        for pol, k in (("L1", 32), ("P2", 32), ("P3", 32)):
            rows = []
            for c in ext:
                src = RecordedSources(lookup(), dol, c.ctx_atoms, c.truth[256], h2[c.id], "cut")
                r = simulate_policy(c.truth[256], pol, src, k, max_new=256, ks=[x for x in KS_ALL if x <= k])
                rows.append((r.start_rounds, r.start_bytes, r.trans))
            ev = evaluate(rows)
            res["bolmo"][f"{pol}|{n}|K{k}"] = ev
            add_row("Bolmo", f"{pol} K={k}", n, ev)
    # OLMo 2 text under the byte rule (restart state = dolma2 token end) and its token rule
    olmo = json.loads((ROOT / "results/drafting/truth/olmo-bf16.json").read_text(encoding="utf-8"))["prompts"]
    for n in NAMES:
        h2 = pickle.load(open(ROOT / f"results/diagnostics/drafts/{n}-healed2.pkl", "rb"))["olmo"]
        rows = []
        tok_rows = []
        for c in ext:
            o = olmo[c.id]
            toks = o["tokens"]["72"]
            prompt2 = o["text"].encode("utf-8") + dol.id_bytes(toks[0])
            oout, ends = b"", []
            for t in toks[1:]:
                oout += dol.id_bytes(t)
                ends.append(len(oout))
            oflags = [0] * len(oout)
            for e in ends:
                oflags[e - 1] = 1
            atoms = [make_atom(prompt2[-1], 1)] + atoms_with_flags(oout, oflags)
            bnd = {0, *ends}
            recs = replay_byte_rounds(atoms, PrecomputedLMDrafter(h2[c.id], dol, "cut"), 32, ctx_atoms=[], multi_patch=True, max_patches=8)
            sr, sb, tr = {0: 0, 1: 0}, {0: 0, 1: 0}, {}
            for r in recs:
                s = int(r.i in bnd)
                nx = int((r.i + r.emitted) in bnd)
                sr[s] += 1
                sb[s] += r.emitted
                tr[(s, nx)] = tr.get((s, nx), 0) + 1
            rows.append((sr, sb, tr))
        ev = evaluate(rows)
        res["olmo"][f"byte_rule|{n}"] = ev
        add_row("OLMo 2 text, byte rule", "L1 K=32", n, ev)
    md += ["", "OLMo 2's own token rule has a single state (every round restarts on a token end); the model is exact there and not tabulated.", ""]

    # Analytic figure: bytes per round against the hit rate at boundaries. Parameters from the best Bolmo LM cell (L1 healed2 360M).
    ev = res["bolmo"]["L1|smollm2-360m|K32"]
    e_mid, e_b = ev["e_state"]["mid"], ev["e_state"]["boundary"]
    h_mid = (e_mid - 1) / e_mid
    q_byte = ev["share_boundary_rounds"]
    md += ["### Analytic figure (SmolLM2-360M healed2 parameters)", "",
           f"Measured: mid-token rounds emit {e_mid:.2f} B (per-byte hit rate {h_mid:.2f}), {q_byte * 100:.0f}% of byte rounds restart at a boundary. "
           f"Free unit = 1 byte (restart at a boundary with probability {q_byte:.2f}) against 1 token = 4 bytes (restart always at a boundary). "
           "Bytes per round for the hit rate at a boundary h_B, mid-token hit rate fixed at the measured value:", "",
           "| h_B | byte free unit | token free unit | ratio |", "|---:|---:|---:|---:|"]
    fig = []
    for hb in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        b_ = two_state_figure(hb, h_mid, q_byte, 1, 1)
        t_ = two_state_figure(hb, h_mid, 1.0, 4, 4)
        fig.append({"h_boundary": hb, "byte": b_, "token": t_})
        md.append(f"| {hb:.2f} | {b_:.2f} | {t_:.2f} | {t_ / b_:.2f} |")
    res["figure"] = {"h_mid": h_mid, "q_byte": q_byte, "rows": fig}
    md.append("")
    (OUT / "twostate.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (OUT / "twostate.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
