"""Drafter diagnostics, analyses 2-5 (offline; CPU only): replays recorded drafter continuations.

Usage: python tools/diag_analyze.py [--names smollm2-360m,qwen2.5-0.5b,smollm2-135m]
Needs results/crossing/b0/drafts/<name>.pkl (Task B, Bolmo text at every byte + OLMo text at token boundaries) and
results/diagnostics/drafts/<name>-olmo-bytes.pkl (tools/diag_record_olmo_bytes.py, OLMo text at every byte).
Output: results/diagnostics/analysis_<name>.json (summarized by tools/diag_report.py).

Texts: "bolmo" = Bolmo's recorded bf16 output (256 new bytes), "olmo" = OLMo 2's greedy output (t0 + 71 tokens),
extended prompt set (25 prompts) as in Task B. Rules: "byte" = Bolmo's (K = 32 bytes, byte and flag must match,
at most 8 patches, one model-predicted byte per round); "token" = OLMo's (K = 8 tokens, one model-predicted token).
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from itertools import accumulate
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from byteprof.drafting import byte_of, make_atom  # noqa: E402
from byteprof.drafting.diagnostics import (BYTE_ENDS, TOKEN_ENDS, atoms_with_flags, classify_position,  # noqa: E402
                                           first_token_bytes, next_token_agreement, per_round, replay_byte_rounds,
                                           replay_token_rounds, retokenize, token_truth, ws_near_miss)
from byteprof.drafting.flags import Dolma2, spans_to_flags  # noqa: E402
from byteprof.drafting.lmdraft import PrecomputedLMDrafter, PrecomputedTokenDrafter  # noqa: E402
from byteprof.drafting.truthio import load_truth, repetition_rate  # noqa: E402

K_BYTES, K_TOK, MAX_PATCHES, N_BYTES = 32, 8, 8, 256
# best cell per drafter in Task B (context mode); both flag protocols are reported
CTX_MODE = {"smollm2-135m": "healed", "smollm2-360m": "raw", "qwen2.5-0.5b": "healed"}
DRAFTS = ROOT / "results/crossing/b0/drafts"
DIAG = ROOT / "results/diagnostics"


class NoFlags:
    """Wraps a byte drafter and strips the boundary flags of its drafts (bytes-only matching)."""

    def __init__(self, inner):
        self.inner = inner

    def reset(self, ctx, truth=None):
        self.inner.reset([a & 255 for a in ctx], None if truth is None else [a & 255 for a in truth])

    def propose(self, k):
        return [a & 255 for a in self.inner.propose(k)]

    def push(self, atoms):
        self.inner.push([a & 255 for a in atoms])


def pool(rounds_by_prompt: dict[str, list], types: dict[str, str]) -> dict:
    allr = [r for rs in rounds_by_prompt.values() for r in rs]
    out = per_round(allr)
    out["ends"] = {}
    for r in allr:
        out["ends"][r.end] = out["ends"].get(r.end, 0) + 1
    by_t: dict[str, list] = defaultdict(list)
    for pid, rs in rounds_by_prompt.items():
        by_t[types[pid]].extend(rs)
    out["by_type"] = {t: per_round(rs) for t, rs in by_t.items()}
    out["ends_by_type"] = {t: {k: sum(r.end == k for r in rs) for k in sorted({r.end for r in rs})} for t, rs in by_t.items()}
    out["L_sum"] = sum(r.L for r in allr)
    out["acc_bytes"] = sum(r.extra.get("acc_bytes", r.L) for r in allr)
    return out


def held_back(ctx: bytes, cont) -> bytes:
    """The bytes of `ctx` that a recorded continuation re-generates first (token healing)."""
    return ctx[len(ctx) - cont.back:] if (cont is not None and cont.back) else b""


def next_byte(cont, ctx: bytes):
    """The drafter's first predicted byte after `ctx`, or None if there is no usable continuation."""
    if cont is None or not cont.cont:
        return None
    back = held_back(ctx, cont)
    if not cont.cont.startswith(back):
        return None
    return cont.cont[len(back):][:1] or None


def hit_rate(rounds) -> float:
    """Per-unit hit rate if a round ended at the first miss: accepted / (accepted + rounds that ended on a miss)."""
    acc = sum(r.L for r in rounds)
    miss = sum(r.end in ("wrong_byte", "wrong_flag", "wrong_token", "patch_end") for r in rounds)
    return acc / (acc + miss) if acc + miss else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="smollm2-360m,qwen2.5-0.5b,smollm2-135m")
    ap.add_argument("--mode", default="", help="raw | healed | healed2 (default: the drafter's Task B cell)")
    a = ap.parse_args()
    dol = Dolma2()
    cases = load_truth(ROOT / "results/drafting/truth/bf16.json")
    olmo = json.loads((ROOT / "results/drafting/truth/olmo-bf16.json").read_text(encoding="utf-8"))["prompts"]
    ids = [p for p, c in cases.items() if c.set != "default"]  # extended set, as in Task B
    types = {p: cases[p].type for p in cases}
    tok_nbytes = lambda t: len(dol.id_bytes(t))  # noqa: E731

    # --- reference texts (independent of the drafter) ---
    texts: dict[str, dict] = {}
    for pid in ids:
        c = cases[pid]
        full_prompt = c.text.encode("utf-8")
        assert c.prompt_bytes + bytes([byte_of(c.truth[N_BYTES][0])]) == full_prompt, pid
        out = bytes(byte_of(x) for x in c.truth[N_BYTES][1:])
        data = full_prompt + out
        tt = token_truth(dol, data, len(full_prompt))
        ids_all, spans_all, used = retokenize(dol, data)
        flags_all = spans_to_flags(spans_all, len(data))
        # Bolmo text with dolma2 flags instead of Bolmo's own (a_0 = last prompt byte)
        atoms_d = atoms_with_flags(data[len(full_prompt) - 1:], flags_all[len(full_prompt) - 1:])
        o = olmo[pid]
        toks = o["tokens"]["72"]
        prompt2 = o["text"].encode("utf-8") + dol.id_bytes(toks[0])
        oout = b""
        ends = []
        for t in toks[1:]:
            oout += dol.id_bytes(t)
            ends.append(len(oout))
        oflags = [0] * len(oout)
        for e in ends:
            oflags[e - 1] = 1
        o_atoms = [make_atom(prompt2[-1], 1)] + atoms_with_flags(oout, oflags)
        bnd_b = {0, *(e - len(full_prompt) for _, e in tt["spans"])}
        bnd_o = {0, *accumulate(len(dol.id_bytes(t)) for t in toks[1:])}
        texts[pid] = {"bnd_bolmo": bnd_b, "bnd_olmo": bnd_o, "full_prompt": full_prompt, "bolmo_out": out, "bolmo_data": data, "tt": tt, "atoms_dolma": atoms_d,
                      "olmo_toks": toks, "olmo_prompt2": prompt2, "olmo_out": oout, "olmo_atoms": o_atoms}

    for name in [x for x in a.names.split(",") if x]:
        with open(DRAFTS / f"{name}.pkl", "rb") as f:
            rec = pickle.load(f)
        with open(DIAG / "drafts" / f"{name}-olmo-bytes.pkl", "rb") as f:
            ob = pickle.load(f)["cases"]
        cm = a.mode or CTX_MODE[name]
        h2 = None
        if cm == "healed2":
            with open(DIAG / "drafts" / f"{name}-healed2.pkl", "rb") as f:
                h2 = pickle.load(f)
        res: dict = {"name": name, "ctx_mode": cm, "cells": {}, "agree": {}, "miss": {}}
        items_bolmo = {pid: (h2["bolmo"][pid] if h2 else rec["bolmo"][pid][cm]) for pid in ids}
        items_olmob = {pid: (h2["olmo"][pid] if h2 else ob[pid][cm]) for pid in ids}  # OLMo text, every byte
        items_olmot = {}  # OLMo text at its token boundaries (the same contexts as the byte recording)
        n_diff = 0
        for pid in ids:
            cum = [0, *accumulate(len(dol.id_bytes(t)) for t in texts[pid]["olmo_toks"][1:])]
            items_olmot[pid] = [items_olmob[pid][e] for e in cum]
            if not h2:  # consistency check against Task B's token-boundary recording
                n_diff += sum(x != y for x, y in zip(items_olmot[pid], rec["olmo"][pid][cm]))
        res["olmo_item_mismatches_vs_task_b"] = n_diff
        print(name, cm, "OLMo token-boundary items differing from Task B's recording:", n_diff, flush=True)
        for fm in ("cut", "full"):
            cell: dict = {}
            rounds: dict[str, dict[str, list]] = defaultdict(dict)
            for pid in ids:
                c, T = cases[pid], texts[pid]
                items_b = items_bolmo[pid]
                # Bolmo text, byte rule with Bolmo's own flags (Task B's cell)
                rounds["bolmo|byte"][pid] = replay_byte_rounds(
                    c.truth[N_BYTES], PrecomputedLMDrafter(items_b, dol, fm), K_BYTES, ctx_atoms=c.ctx_atoms,
                    multi_patch=True, max_patches=MAX_PATCHES)
                # Bolmo text, byte rule with dolma2 flags in the reference
                rounds["bolmo|byte_dolma"][pid] = replay_byte_rounds(
                    T["atoms_dolma"], PrecomputedLMDrafter(items_b, dol, fm), K_BYTES, ctx_atoms=c.ctx_atoms,
                    multi_patch=True, max_patches=MAX_PATCHES)
                # Bolmo text, bytes only (no flag has to match)
                rounds["bolmo|byte_noflag"][pid] = replay_byte_rounds(
                    [x & 255 for x in c.truth[N_BYTES]], NoFlags(PrecomputedLMDrafter(items_b, dol, fm)), K_BYTES,
                    ctx_atoms=c.ctx_atoms, multi_patch=True)
                # what-if: the bonus completes the token (rounds start at token boundaries), byte-level matching
                rounds["bolmo|byte_aligned"][pid] = replay_byte_rounds(
                    c.truth[N_BYTES], PrecomputedLMDrafter(items_b, dol, fm), K_BYTES, ctx_atoms=c.ctx_atoms,
                    multi_patch=True, max_patches=MAX_PATCHES, align=T["bnd_bolmo"])
                # Bolmo text, token rule (dolma2 retokenization is the reference)
                tt = T["tt"]
                P = len(T["full_prompt"])
                items_t = [items_b[e - P] for _, e in tt["spans"]]
                d = PrecomputedTokenDrafter(items_t, dol)
                d.set_text(T["bolmo_data"][:tt["spans"][0][0]], T["bolmo_data"][tt["spans"][0][0]:tt["spans"][0][1]])
                rounds["bolmo|token"][pid] = replay_token_rounds(tt["ids"], d, K_TOK, tok_nbytes)
                # OLMo text, byte rule (flags = OLMo's token ends) and bytes only
                items_o = items_olmob[pid]
                rounds["olmo|byte"][pid] = replay_byte_rounds(
                    T["olmo_atoms"], PrecomputedLMDrafter(items_o, dol, fm), K_BYTES, ctx_atoms=[], multi_patch=True,
                    max_patches=MAX_PATCHES)
                rounds["olmo|byte_noflag"][pid] = replay_byte_rounds(
                    [x & 255 for x in T["olmo_atoms"]], NoFlags(PrecomputedLMDrafter(items_o, dol, fm)), K_BYTES,
                    ctx_atoms=[], multi_patch=True)
                rounds["olmo|byte_aligned"][pid] = replay_byte_rounds(
                    T["olmo_atoms"], PrecomputedLMDrafter(items_o, dol, fm), K_BYTES, ctx_atoms=[], multi_patch=True,
                    max_patches=MAX_PATCHES, align=T["bnd_olmo"])
                # OLMo text, token rule (Task B's OLMo cell)
                o = rec["olmo"][pid]
                d = PrecomputedTokenDrafter(items_olmot[pid], dol)
                d.set_text(o["prompt_bytes"], dol.id_bytes(o["tokens"][0]))
                rounds["olmo|token"][pid] = replay_token_rounds(o["tokens"], d, K_TOK, tok_nbytes)
            for key, rr in rounds.items():
                cell[key] = pool(rr, types)
                cell[key]["hit_rate"] = hit_rate([r for rs in rr.values() for r in rs])
            res["cells"][fm] = cell
            print(name, fm, {k: round(v["bytes"], 2) for k, v in cell.items()}, flush=True)
            if fm == "cut":
                # first-miss positions (analysis 5), byte rule, wrong-byte rounds
                for text in ("bolmo", "olmo"):
                    cnt = defaultdict(int)
                    near = tot = first_byte = 0
                    for pid in ids:
                        T = texts[pid]
                        if text == "bolmo":
                            data, off = T["bolmo_data"], len(T["full_prompt"]) - 1
                        else:
                            data, off = T["olmo_prompt2"] + T["olmo_out"], len(T["olmo_prompt2"]) - 1
                        for r in rounds[f"{text}|byte"][pid]:
                            if r.end != "wrong_byte":
                                continue
                            tot += 1
                            cnt[classify_position(data, off + r.miss_i)] += 1
                            near += ws_near_miss(r.miss_draft, r.miss_truth)
                            first_byte += r.L == 0
                    st = {"boundary": [0, 0], "mid": [0, 0]}  # rounds, emitted bytes by where the round starts
                    for pid in ids:
                        for r in rounds[f"{text}|byte"][pid]:
                            k_ = "boundary" if r.i in texts[pid][f"bnd_{text}"] else "mid"
                            st[k_][0] += 1
                            st[k_][1] += r.emitted
                    res["round_start"] = res.get("round_start", {})
                    res["round_start"][text] = {k_: {"rounds": v[0], "bytes_per_round": v[1] / max(v[0], 1)} for k_, v in st.items()}
                    res["miss"][text] = {"rounds": tot, "positions": dict(cnt), "ws_near_miss": near, "first_byte_miss": first_byte}

        # --- analysis 4: next-token agreement at dolma2 token boundaries, both texts ---
        for text in ("bolmo", "olmo"):
            by_t: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # n, prefix-agree, exact
            by_ctx = {"boundary": [0, 0], "mid": [0, 0]}  # next-byte agreement at all positions
            for pid in ids:
                T = texts[pid]
                if text == "bolmo":
                    tt = T["tt"]
                    P = len(T["full_prompt"])
                    items = items_bolmo[pid]
                    toks = [T["bolmo_data"][s:e] for s, e in tt["spans"]]
                    for j in range(len(toks) - 1):
                        it_ = items[tt["spans"][j][1] - P]
                        fb = first_token_bytes(it_, held_back(T["bolmo_data"][:tt["spans"][j][1]], it_))
                        ok, ex = next_token_agreement(fb, toks[j + 1])
                        s = by_t[types[pid]]
                        s[0] += 1
                        s[1] += ok
                        s[2] += ex
                    bnd = {e - P for _, e in tt["spans"]}
                    for i in range(0, len(T["bolmo_out"])):
                        it = items[i]
                        nb_ = next_byte(it, T["bolmo_data"][:len(T["full_prompt"]) + i])
                        if nb_ is None:
                            continue
                        key = "boundary" if i in bnd else "mid"
                        by_ctx[key][0] += 1
                        by_ctx[key][1] += nb_ == T["bolmo_out"][i:i + 1]
                else:
                    o = rec["olmo"][pid]
                    cum_ = [0, *accumulate(len(dol.id_bytes(t)) for t in o["tokens"][1:])]
                    items = items_olmot[pid]
                    toks = o["tokens"]
                    ctx_b = T["olmo_prompt2"] + T["olmo_out"]
                    for j in range(len(toks) - 1):
                        fb = first_token_bytes(items[j], held_back(ctx_b[:len(T["olmo_prompt2"]) + cum_[j]], items[j]))
                        ok, ex = next_token_agreement(fb, dol.id_bytes(toks[j + 1]))
                        s = by_t[types[pid]]
                        s[0] += 1
                        s[1] += ok
                        s[2] += ex
                    bnd = set(cum_)
                    for i in range(0, len(T["olmo_out"])):
                        nb_ = next_byte(items_olmob[pid][i], T["olmo_prompt2"] + T["olmo_out"][:i])
                        if nb_ is None:
                            continue
                        key = "boundary" if i in bnd else "mid"
                        by_ctx[key][0] += 1
                        by_ctx[key][1] += nb_ == T["olmo_out"][i:i + 1]
            tot = [sum(v[i] for v in by_t.values()) for i in range(3)]
            res["agree"][text] = {"all": {"n": tot[0], "prefix": tot[1] / tot[0], "exact": tot[2] / tot[0]},
                                  "by_type": {t: {"n": v[0], "prefix": v[1] / v[0], "exact": v[2] / v[0]} for t, v in by_t.items()},
                                  "next_byte": {k: {"n": v[0], "agree": v[1] / max(v[0], 1)} for k, v in by_ctx.items()}}
            print(name, "agree", text, round(tot[1] / tot[0], 3), round(tot[2] / tot[0], 3), res["agree"][text]["next_byte"], flush=True)
        (DIAG / f"analysis_{name}_{cm}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    # --- repetition (analysis 6), independent of the drafter ---
    rep = {}
    for text in ("bolmo", "olmo"):
        r = t = 0
        for pid in ids:
            T = texts[pid]
            if text == "bolmo":
                x, y = repetition_rate(T["full_prompt"], T["bolmo_out"])
            else:
                x, y = repetition_rate(T["olmo_prompt2"], T["olmo_out"])
            r += x
            t += y
        rep[text] = {"repeated": r, "windows": t, "rate": r / t}
    (DIAG / "repetition.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("repetition", rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
