"""Task B, Stage B0.1 (H34): Bolmo's emitted boundary flags against the dolma2 tokenization of its own output.

Usage: python tools/cross_flag_agreement.py [--truth results/drafting/truth/bf16.json]
Output: results/crossing/b0/flags.{md,json}
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from byteprof.drafting.flags import Dolma2, classify_diff, compare_flags  # noqa: E402
from byteprof.drafting.truthio import load_truth  # noqa: E402

TYPES = ("prose_en", "code_py", "prose_de", "json", "unicode_mix")


class Acc:
    def __init__(self) -> None:
        self.pos = self.agree = self.patches = self.exact = 0
        self.per_prompt: list[tuple[float, float]] = []

    def add(self, r: dict) -> None:
        self.pos += r["positions"]
        self.agree += r["agree"]
        self.patches += r["patches"]
        self.exact += r["patches_exact"]
        if r["positions"]:
            self.per_prompt.append((r["agree"] / r["positions"], r["patches_exact"] / max(r["patches"], 1)))

    def row(self) -> dict:
        return {"positions": self.pos, "agree": self.agree, "byte_agreement": self.agree / max(self.pos, 1),
                "patches": self.patches, "patches_exact": self.exact, "patch_exact": self.exact / max(self.patches, 1),
                "min_prompt_byte": min((a for a, _ in self.per_prompt), default=0.0),
                "min_prompt_patch": min((b for _, b in self.per_prompt), default=0.0)}


def run(truth_path: str, shifts=(-1, 0, 1)) -> dict:
    dol = Dolma2()
    cases = load_truth(truth_path)
    out: dict = {"truth": str(truth_path), "cells": {}, "kinds": {}, "examples": {}, "shift": {}, "prompt_mask": {}}
    for n in (256, 1024):
        by_type: dict[str, Acc] = defaultdict(Acc)
        by_set: dict[str, Acc] = defaultdict(Acc)
        kinds: Counter = Counter()
        ex: dict[str, list] = defaultdict(list)
        shift_acc = {s: Acc() for s in shifts}
        pm = Acc()
        for pid, c in cases.items():
            atoms = c.truth.get(n)
            if not atoms:
                continue
            ctx_flags = [a >> 8 for a in c.ctx_atoms]
            data = c.prompt_bytes + bytes(a & 255 for a in atoms)
            flags = ctx_flags + [a >> 8 for a in atoms]
            assert len(flags) == len(data)
            spans, covered = dol.token_spans(data)
            first = len(ctx_flags)
            r = compare_flags(flags, spans, first, covered)
            by_type[c.type].add(r)
            by_type["all"].add(r)
            by_set[c.set].add(r)
            for p, bf, rf in r["diffs"]:
                k = classify_diff(data, p, bf)
                kinds[k] += 1
                if len(ex[k]) < 4:
                    ex[k].append({"prompt": pid, "pos": p, "context": data[max(0, p - 12):p + 1].decode("utf-8", "replace"),
                                  "next": data[p + 1:p + 6].decode("utf-8", "replace")})
            for s in shifts:  # which alignment fits: flag on the last byte of a token (s = 0), or shifted by s
                sh = flags[s:] + [0] * s if s >= 0 else [0] * (-s) + flags[:s]
                shift_acc[s].add(compare_flags(sh, spans, first, covered))
            # prompt mask (prefill boundary predictor) against the tokenization of the prompt alone
            ps, pc = dol.token_spans(c.prompt_bytes)
            pm.add(compare_flags(ctx_flags, ps, 0, pc))
        out["cells"][str(n)] = {t: by_type[t].row() for t in (*TYPES, "all") if t in by_type}
        for name, acc in by_set.items():
            out["cells"][str(n)]["set:" + name] = acc.row()
        out["kinds"][str(n)] = dict(kinds.most_common())
        out["examples"][str(n)] = ex
        out["shift"][str(n)] = {str(s): shift_acc[s].row() for s in shifts}
        out["prompt_mask"][str(n)] = pm.row()
    return out


def render(res: dict) -> str:
    L = ["# B0.1: Bolmo flags vs dolma2 tokenization of its own output (H34)", "",
         f"truth: `{res['truth']}`. Evaluated: generated positions a_0 .. a_n (prompt bytes excluded), up to the end of",
         "the second-to-last dolma2 token of prompt + output (the final token may be incomplete). A flag sits on the",
         "last byte of a token/patch. Patch exact = a token with the same start and end.", ""]
    for n, cells in res["cells"].items():
        L += [f"## {n} new bytes", "", "| prompts | positions | byte agreement | patches | patch exact | worst prompt (byte / patch) |",
              "|---|---:|---:|---:|---:|---:|"]
        for t, r in cells.items():
            if not r:
                continue
            L.append(f"| {t} | {r['positions']} | {r['byte_agreement']:.4f} | {r['patches']} | {r['patch_exact']:.4f} | "
                     f"{r['min_prompt_byte']:.3f} / {r['min_prompt_patch']:.3f} |")
        L += ["", "Disagreement kinds (direction/kind; `extra` = Bolmo cuts where dolma2 does not):", "",
              "| kind | positions | share |", "|---|---:|---:|"]
        tot = sum(res["kinds"][n].values())
        for k, v in res["kinds"][n].items():
            L.append(f"| {k} | {v} | {v / max(tot, 1):.3f} |")
        L += ["", "Examples (text up to the disputed byte | next bytes):", ""]
        for k, es in res["examples"][n].items():
            for e in es[:2]:
                L.append(f"- {k}: `{e['prompt']}` ...`{e['context']!r}` | `{e['next']!r}`")
        L += ["", "Alignment check (flags shifted by s bytes against the token ends; s = 0 is the convention used):", "",
              "| shift s | byte agreement | patch exact |", "|---:|---:|---:|"]
        for s, r in res["shift"][n].items():
            L.append(f"| {s} | {r['byte_agreement']:.4f} | {r['patch_exact']:.4f} |")
        pm = res["prompt_mask"][n]
        L += ["", f"Prompt mask (prefill boundary predictor) against the tokenization of the prompt alone: "
                  f"byte agreement {pm['byte_agreement']:.4f}, patch exact {pm['patch_exact']:.4f} "
                  f"({pm['positions']} positions).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=str(ROOT / "results/drafting/truth/bf16.json"))
    ap.add_argument("--out", default=str(ROOT / "results/crossing/b0"))
    a = ap.parse_args()
    res = run(a.truth)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "flags.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (od / "flags.md").write_text(render(res), encoding="utf-8")
    print(render(res))
