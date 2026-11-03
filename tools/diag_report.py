"""Drafter diagnostics: tables (markdown) from results/diagnostics/analysis_<name>_<mode>.json, ce_*.json, repetition.json.

Usage: python tools/diag_report.py   ->  results/diagnostics/report.md
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results/diagnostics"
NAMES = ("smollm2-135m", "smollm2-360m", "qwen2.5-0.5b")
MODES = ("raw", "healed", "healed2")
BEST = {"smollm2-135m": "healed", "smollm2-360m": "raw", "qwen2.5-0.5b": "healed"}  # Task B's cell
ENDS_B = ("wrong_byte", "wrong_flag", "exhausted", "cap", "patch_cap", "end_of_generation")
ENDS_T = ("wrong_token", "exhausted", "cap", "end_of_generation")


def load(name, mode):
    p = D / f"analysis_{name}_{mode}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f2(x):
    return f"{x:.2f}"


def pct(x):
    return f"{100 * x:.1f}%"


def main() -> int:
    out = []
    A = {(n, m): load(n, m) for n in NAMES for m in MODES}

    out.append("### Grid: bytes emitted per round, same drafter continuations, extended set (25 prompts)\n")
    out.append("K = 32 bytes (byte rule), 8 tokens (token rule). `cut` = the task's flag protocol. Byte rule = Bolmo's (byte and flag must match, one model byte per round); token rule = OLMo's (token must match, one model token per round).\n")
    out.append("| drafter | context | flags | Bolmo text, byte rule | Bolmo text, token rule | OLMo text, byte rule | OLMo text, token rule | token/byte on Bolmo text | token/byte on OLMo text | OLMo-text/Bolmo-text, byte rule | same, token rule |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in NAMES:
        for m in MODES:
            a = A[(n, m)]
            if not a:
                continue
            for fm in ("cut", "full"):
                c = a["cells"][fm]
                bb, bt, ob, ot = (c[k]["bytes"] for k in ("bolmo|byte", "bolmo|token", "olmo|byte", "olmo|token"))
                out.append(f"| {n} | {m}{' (Task B)' if BEST[n] == m else ''} | {fm} | {f2(bb)} | {f2(bt)} | {f2(ob)} | {f2(ot)} | {bt / bb:.2f}x | {ot / ob:.2f}x | {ob / bb:.2f}x | {ot / bt:.2f}x |")

    out.append("\n### Ladder on the same text (bytes per round; `cut`): from Bolmo's rule to OLMo's rule\n")
    out.append("byte+flags -> bytes only (flags need not match) -> bytes only with the bonus completing the token (`aligned`, what-if) -> token rule without its bonus token (accepted tokens only) -> token rule (OLMo's, with bonus token).\n")
    out.append("| drafter / context | text | byte+flags | bytes only | aligned byte (flags) | token, accepted only | token rule | bonus token bytes/round |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for n in NAMES:
        for m in MODES:
            a = A[(n, m)]
            if not a:
                continue
            c = a["cells"]["cut"]
            for text in ("bolmo", "olmo"):
                t = c[f"{text}|token"]
                acc = t["acc_bytes"] / t["rounds"]
                out.append(f"| {n} / {m} | {text} | {f2(c[f'{text}|byte']['bytes'])} | {f2(c[f'{text}|byte_noflag']['bytes'])} | {f2(c[f'{text}|byte_aligned']['bytes'])} | {f2(acc)} | {f2(t['bytes'])} | {f2(t['bytes'] - acc)} |")

    out.append("\n### Why rounds end (shares of rounds; `cut`)\n")
    out.append("| drafter / context | text, rule | rounds | " + " | ".join(ENDS_B) + " | hit rate p | p/(1-p) |")
    out.append("|---|---|---:|" + "---:|" * (len(ENDS_B) + 2))
    for n in NAMES:
        for m in MODES:
            a = A[(n, m)]
            if not a:
                continue
            c = a["cells"]["cut"]
            for key in ("bolmo|byte", "olmo|byte"):
                v = c[key]
                out.append(f"| {n} / {m} | {key} | {v['rounds']} | " + " | ".join(pct(v['ends'].get(k, 0) / v['rounds']) for k in ENDS_B) + f" | {v['hit_rate']:.3f} | {v['hit_rate'] / (1 - v['hit_rate']):.2f} |")
            for key in ("bolmo|token", "olmo|token"):
                v = c[key]
                out.append(f"| {n} / {m} | {key} (wrong_token, exhausted, cap, end) | {v['rounds']} | " + " | ".join(pct(v['ends'].get(k, 0) / v['rounds']) for k in ENDS_T) + f" | | | {v['hit_rate']:.3f} | {v['hit_rate'] / (1 - v['hit_rate']):.2f} |")

    out.append("\n### Why rounds end, by prompt type (Bolmo text, byte rule, `cut`, Task B's cell per drafter)\n")
    types = ["prose_en", "code_py", "prose_de", "json", "unicode_mix"]
    out.append("| drafter | type | rounds | wrong_byte | wrong_flag | exhausted | cap / patch_cap | bytes/round |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for n in NAMES:
        a = A[(n, BEST[n])]
        v = a["cells"]["cut"]["bolmo|byte"]
        for t in types:
            e = v["ends_by_type"].get(t, {})
            tot = sum(e.values())
            out.append(f"| {n} | {t} | {tot} | {pct(e.get('wrong_byte', 0) / tot)} | {pct(e.get('wrong_flag', 0) / tot)} | {pct(e.get('exhausted', 0) / tot)} | {pct((e.get('cap', 0) + e.get('patch_cap', 0)) / tot)} | {f2(v['by_type'][t]['bytes'])} |")

    out.append("\n### Next-token agreement of the drafter's top-1 token with the model's actual next token (dolma2 token boundaries)\n")
    out.append("Agreement = the drafter's first token and the model's next dolma2 token agree up to the shorter length (the task's definition); exact = identical bytes. Next byte = the drafter's first byte equals the next byte, at all byte positions, split by whether the context ends on a token end.\n")
    out.append("| drafter / context | model's text | n tokens | agree (shorter length) | exact | next byte, ctx at token end | next byte, ctx inside a token |")
    out.append("|---|---|---:|---:|---:|---:|---:|")
    for n in NAMES:
        for m in MODES:
            a = A[(n, m)]
            if not a:
                continue
            for text in ("bolmo", "olmo"):
                g = a["agree"][text]
                nb = g["next_byte"]
                out.append(f"| {n} / {m} | {text} | {g['all']['n']} | {pct(g['all']['prefix'])} | {pct(g['all']['exact'])} | {pct(nb['boundary']['agree'])} (n={nb['boundary']['n']}) | {pct(nb['mid']['agree'])} (n={nb['mid']['n']}) |")
    out.append("\nBy prompt type (Task B's cell per drafter), agree (shorter length):\n")
    out.append("| drafter | text | " + " | ".join(types) + " |")
    out.append("|---|---|" + "---:|" * len(types))
    for n in NAMES:
        a = A[(n, BEST[n])]
        for text in ("bolmo", "olmo"):
            g = a["agree"][text]["by_type"]
            out.append(f"| {n} | {text} | " + " | ".join(pct(g[t]["prefix"]) if t in g else "" for t in types) + " |")

    out.append("\n### Where the first miss falls (byte rule, rounds ending on a wrong byte, `cut`)\n")
    out.append("| drafter / context | text | wrong-byte rounds | on the first draft byte | word start | inside a word | whitespace/punct. | multi-byte char | whitespace-only near miss |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for n in NAMES:
        for m in MODES:
            a = A[(n, m)]
            if not a:
                continue
            for text in ("bolmo", "olmo"):
                x = a["miss"][text]
                tot = x["rounds"]
                p = x["positions"]
                out.append(f"| {n} / {m} | {text} | {tot} | {pct(x['first_byte_miss'] / tot)} | " + " | ".join(pct(p.get(k, 0) / tot) for k in ("word_start", "inside_word", "whitespace_punct", "multibyte")) + f" | {pct(x['ws_near_miss'] / tot)} |")

    out.append("\n### Byte rule by where the round starts (Bolmo text / OLMo text; `cut`)\n")
    out.append("| drafter / context | text | rounds starting on a token end | bytes/round there | rounds starting inside a token | bytes/round there |")
    out.append("|---|---|---:|---:|---:|---:|")
    for n in NAMES:
        for m in MODES:
            a = A[(n, m)]
            if not a:
                continue
            for text in ("bolmo", "olmo"):
                r = a["round_start"][text]
                out.append(f"| {n} / {m} | {text} | {r['boundary']['rounds']} ({pct(r['boundary']['rounds'] / (r['boundary']['rounds'] + r['mid']['rounds']))}) | {f2(r['boundary']['bytes_per_round'])} | {r['mid']['rounds']} | {f2(r['mid']['bytes_per_round'])} |")

    out.append("\n### Text predictability\n")
    rp = json.loads((D / "repetition.json").read_text(encoding="utf-8"))
    out.append(f"Repetition (8-byte windows of the output seen earlier in prompt + output): Bolmo {pct(rp['bolmo']['rate'])} ({rp['bolmo']['repeated']}/{rp['bolmo']['windows']}), OLMo 2 {pct(rp['olmo']['rate'])} ({rp['olmo']['repeated']}/{rp['olmo']['windows']}).\n")
    out.append("| drafter | Bolmo text, bits/byte | OLMo text, bits/byte | " + " | ".join(f"{t} (B / O)" for t in types) + " |")
    out.append("|---|---:|---:|" + "---:|" * len(types))
    for n in NAMES:
        p = D / f"ce_{n}.json"
        if not p.exists():
            continue
        ce = json.loads(p.read_text(encoding="utf-8"))
        bt = ce["by_type"]
        out.append(f"| {n} | {ce['bolmo']['all']['bpb']:.3f} | {ce['olmo']['all']['bpb']:.3f} | " + " | ".join(f"{bt[f'bolmo|{t}']:.2f} / {bt[f'olmo|{t}']:.2f}" for t in types) + " |")

    (D / "report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
