"""Build time, memory and per-step time of the Bolmo subword automaton (Stage 1, item 4).

    python tools/automaton_stats.py
"""

import json
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer  # noqa: E402

from byteprof.optim.subword_automaton import SubwordAutomaton  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
tok = AutoTokenizer.from_pretrained("allenai/Bolmo-1B", revision="452690a8cfab1dd1ec27b74f2667632f17a21b46",
                                    trust_remote_code=True, local_files_only=True)

tracemalloc.start()
t0 = time.perf_counter()
auto = SubwordAutomaton.from_tokenizer(tok)
build_s = time.perf_counter() - t0
cur, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"entries={auto.n_entries} states={auto.n_states} build={build_s:.2f}s "
      f"memory_after_build={cur / 2**20:.0f} MiB peak_during_build={peak / 2**20:.0f} MiB")

texts = json.loads((ROOT / "tests/data/bolmo_generated.json").read_text(encoding="utf-8"))
ids = [i for t in texts for i in tok(t)["input_ids"]]
for label in ("cold (memo empty)", "warm (memo filled)"):
    state = 0
    t0 = time.perf_counter()
    for b in ids:
        state, _ = auto.step(state, b)
    dt = time.perf_counter() - t0
    print(f"step() {label}: {1e6 * dt / len(ids):.2f} us/byte over {len(ids)} bytes")

# the reference for comparison: expand_byte_ids(n_last=1) on a growing history
hist = ids[:2000]
t0 = time.perf_counter()
for n in range(1, len(hist) + 1, 20):
    tok.expand_byte_ids(hist[:n], n_last=1)
dt = time.perf_counter() - t0
print(f"reference expand_byte_ids(n_last=1) incl. list copy, history 1..2000: {1e6 * dt / len(range(1, len(hist) + 1, 20)):.1f} us/call")
