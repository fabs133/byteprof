"""Builds the drafting corpus + n-gram tables once and caches them under results/drafting/corpus/.

Writes corpus.bin (the <= 20 MB corpus), flags.bin (1 on the last byte of every dolma2 token), olmo_tokens.npy and
olmo_token_ends.npy (the same corpus as OLMo 2 token ids, for the token n-gram of Stage 3), sources.json and
build.json (times, peak RAM, overlap check). The n-gram tables themselves are rebuilt by `load_ngram()`
(about 30-60 s), they are not cached.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from byteprof.drafting.corpus import build_corpus, segment_with_tokenizer, shared_windows  # noqa: E402
from byteprof.drafting.ngram import NGramModel  # noqa: E402
from byteprof.runner import load_prompts  # noqa: E402

OUT = ROOT / "results" / "drafting" / "corpus"
TOK = next((Path.home() / ".cache/huggingface/hub/models--allenai--OLMo-2-0425-1B/snapshots").glob("*/tokenizer.json"))
ORDERS = (1, 2, 3, 4, 5, 6, 8, 12)


def peak_mb() -> float:
    return psutil.Process().memory_info().peak_wset / 2**20


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    info: dict = {}
    t0 = time.perf_counter()
    corpus = build_corpus()
    data = corpus.data
    info["corpus_build_s"] = time.perf_counter() - t0
    info["corpus_bytes"] = len(data)
    info["sources"] = corpus.sources
    (OUT / "corpus.bin").write_bytes(data)
    (OUT / "sources.json").write_text(json.dumps(corpus.sources, indent=1), encoding="utf-8")
    print("corpus", len(data), "bytes", f"{info['corpus_build_s']:.1f}s", flush=True)

    t0 = time.perf_counter()
    flags, ids, ends = segment_with_tokenizer(corpus, TOK)
    info["segmentation_s"] = time.perf_counter() - t0
    info["tokens"] = int(len(ids))
    info["bytes_per_token"] = len(data) / len(ids)
    info["flag_rate"] = float(flags.mean())
    flags.tofile(OUT / "flags.bin")
    np.save(OUT / "olmo_tokens.npy", ids)
    np.save(OUT / "olmo_token_ends.npy", ends)
    print("segmented", info["tokens"], "tokens", f"{info['segmentation_s']:.1f}s", flush=True)

    # overlap with every prompt (default + extended): no shared 32-byte window
    texts = {}
    for f in ("prompts/default.jsonl", "prompts/drafting_ext.jsonl"):
        for p in load_prompts(ROOT / f):
            texts[f"{Path(f).stem}:{p['id']}"] = p["text"].encode("utf-8")
    t0 = time.perf_counter()
    hits = shared_windows(data, texts, 32)
    info["overlap_check_s"] = time.perf_counter() - t0
    info["shared_32_byte_windows"] = len(hits)
    info["overlapping_prompts"] = sorted({h[0] for h in hits})
    print("overlap", len(hits), info["overlapping_prompts"], flush=True)

    t0 = time.perf_counter()
    model = NGramModel.build(data, flags, ORDERS, bits=24)
    info["ngram_build_s"] = time.perf_counter() - t0
    info["ngram_orders"] = list(ORDERS)
    info["ngram_entries"] = model.n_entries
    info["peak_ram_mb"] = peak_mb()
    print("ngram build", f"{info['ngram_build_s']:.1f}s", "peak RAM", f"{info['peak_ram_mb']:.0f} MB", model.n_entries,
          flush=True)
    (OUT / "build.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
