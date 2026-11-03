"""Writes prompts/drafting_ext.jsonl: 25 prompts, 5 each of English prose, Python, German prose, JSON and
mixed Unicode, for TASK_drafting.md Stage 0. Provenance is stored in the `source` field of every line.

- en_*, de_*, uni_*: written for this task (original text, not copied from any corpus).
- py_*: function openings from third-party packages in the venv's site-packages (not the standard library).
- json_*: the beginning of real JSON files of other models in the local HF cache (config files).
The n-gram corpus is WikiText-2 train + standard-library sources + opus_books German; the overlap test
(tests/test_drafting_corpus.py) checks that no prompt shares a 32-byte window with it.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SP = ROOT / ".venv" / "Lib" / "site-packages"
HUB = Path.home() / ".cache" / "huggingface" / "hub"

EN = [
    "The lighthouse at Cape Verrin was built in 1874 to replace a wooden beacon that had burned down twice in a decade. Its first keeper, a retired naval engineer named",
    "When the committee met to discuss the proposed changes to the water supply, the first question raised was not about cost but about",
    "Photosynthesis is the process by which plants, algae and some bacteria convert light energy into chemical energy. In the first stage,",
    "I had never intended to stay in the village for more than a week, but on the morning the ferry stopped running, I",
    "A good recipe for sourdough bread depends less on exact measurements than on patience. Start by feeding your starter the night before, and",
]
DE = [
    "Der Schwarzwald ist ein Mittelgebirge im Südwesten Deutschlands und gehört zu den beliebtesten Urlaubsregionen des Landes. Besonders bekannt sind",
    "Als ich gestern Abend nach Hause kam, lag ein Brief auf dem Tisch, den ich zuerst gar nicht beachtet habe, weil",
    "Die Bundesregierung hat am Mittwoch ein neues Gesetz zur Förderung erneuerbarer Energien beschlossen. Nach Angaben des Ministeriums sollen",
    "Zutaten für vier Personen: 500 Gramm Mehl, ein halber Liter Milch, drei Eier und eine Prise Salz. Zuerst verrührt man",
    "In der Informatik bezeichnet man als Algorithmus eine eindeutige Handlungsvorschrift zur Lösung eines Problems. Ein Algorithmus besteht aus",
]
UNI = [
    "Übersicht / Overview / 概要 / Обзор: Diese Tabelle zeigt die Preise in €, £ und ¥ für 2026 — price (€): 12,50; prix (€):",
    "Привет, мир! こんにちは世界! مرحبا بالعالم! Hello, world! Ο καιρός σήμερα είναι ηλιόλουστος, και",
    "日本語のテキスト:東京は日本の首都であり、世界最大級の都市圏を形成している。人口は約",
    "Café menu: crème brûlée — 6,50 €, tarte à l’orange — 5,00 €, œufs à la neige — 4,80 €, thé vert 🍵 —",
    "Math: ∑_{i=1}^{n} i = n(n+1)/2, ∀x ∈ ℝ: x² ≥ 0, α + β = γ, ∞ > 10⁶; units: 5 µm, 20 °C, 3 × 10⁸ m/s, and",
]
PY = [  # (file under site-packages, regex of the def line)
    ("packaging/version.py", r"def _parse_letter_version"),
    ("filelock/_api.py", r"def acquire\("),
    ("networkx/algorithms/shortest_paths/generic.py", r"def shortest_path\("),
    ("einops/einops.py", r"def reduce\("),
    ("accelerate/utils/memory.py", r"def find_executable_batch_size"),
]
JSON = [  # (glob under the HF hub dir, max bytes)
    "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*/config.json",
    "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*/speech_tokenizer/config.json",
    "models--openai--whisper-large-v3-turbo/snapshots/*/config.json",
    "models--openai--whisper-large-v3-turbo/snapshots/*/generation_config.json",
    "models--openai--whisper-large-v3-turbo/snapshots/*/preprocessor_config.json",
]
N_CODE = 260
N_JSON = 230


def cut_at_line(text: str, n: int) -> str:
    chunk = text[:n]
    nl = chunk.rfind("\n")
    return chunk[: nl + 1] if nl > n // 2 else chunk


def main() -> int:
    rows = []
    for i, t in enumerate(EN, 1):
        rows.append({"id": f"en_{i}", "type": "prose_en", "source": "written for this task", "text": t})
    for i, (rel, pat) in enumerate(PY, 1):
        src = (SP / rel).read_text(encoding="utf-8")
        m = re.search(pat, src)
        if not m:
            print("no match", rel, pat)
            return 1
        start = src.rfind("\n", 0, m.start()) + 1
        rows.append({"id": f"py_{i}", "type": "code_py", "source": f"site-packages/{rel} (from `{pat}`)",
                     "text": cut_at_line(src[start:], N_CODE)})
    for i, t in enumerate(DE, 1):
        rows.append({"id": f"de_{i}", "type": "prose_de", "source": "written for this task", "text": t})
    for i, g in enumerate(JSON, 1):
        files = sorted(HUB.glob(g))
        if not files:
            print("no file", g)
            return 1
        txt = files[0].read_text(encoding="utf-8")
        rows.append({"id": f"json_{i}", "type": "json", "source": f"HF hub {g.split('/')[0]} / {g.split('/', 3)[-1]}",
                     "text": cut_at_line(txt, N_JSON).rstrip("\n") if "\n" in txt[:N_JSON] else txt[:N_JSON]})
    for i, t in enumerate(UNI, 1):
        rows.append({"id": f"uni_{i}", "type": "unicode_mix", "source": "written for this task", "text": t})
    out = ROOT / "prompts" / "drafting_ext.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    for r in rows:
        print(r["id"], len(r["text"].encode("utf-8")), repr(r["text"][:70]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
