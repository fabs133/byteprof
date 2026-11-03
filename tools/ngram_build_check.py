"""Builds the n-gram tables from the cached corpus in a fresh process; reports time and peak RAM."""
import sys, time
from pathlib import Path
import numpy as np, psutil
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from byteprof.drafting.ngram import NGramModel
C = ROOT / "results" / "drafting" / "corpus"
data = (C / "corpus.bin").read_bytes()
flags = np.fromfile(C / "flags.bin", np.uint8)
base = psutil.Process().memory_info().rss / 2**20
t0 = time.perf_counter()
m = NGramModel.build(data, flags, (1, 2, 3, 4, 5, 6, 8, 12), bits=24, shards=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
print(f"build {time.perf_counter()-t0:.1f}s peak_ram {psutil.Process().memory_info().peak_wset/2**20:.0f} MB rss_after {psutil.Process().memory_info().rss/2**20:.0f} MB (before build {base:.0f} MB)")
print(m.n_entries)
