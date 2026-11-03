"""The n-gram corpus must not contain any prompt text: no shared 32-byte window (TASK_drafting.md)."""

from pathlib import Path

import pytest

from byteprof.drafting import corpus as C
from byteprof.runner import load_prompts

ROOT = Path(__file__).resolve().parent.parent
HAVE_DATA = all(
    (C.HF_CACHE / ("datasets--" + repo.replace("/", "--")) / "snapshots" / rev / f).is_file()
    for repo, rev, f in (("Salesforce/wikitext", C.WIKITEXT_REV, C.WIKITEXT_TRAIN),
                         ("Helsinki-NLP/opus_books", C.OPUS_REV, C.OPUS_FILE)))


def test_prompt_files_have_the_documented_shape():
    ext = load_prompts(ROOT / "prompts" / "drafting_ext.jsonl")
    assert len(ext) == 25
    kinds = {}
    for p in ext:
        kinds.setdefault(p["id"].split("_")[0], []).append(p)
    assert {k: len(v) for k, v in kinds.items()} == {"en": 5, "py": 5, "de": 5, "json": 5, "uni": 5}


@pytest.mark.skipif(not HAVE_DATA, reason="WikiText-2 train / opus_books not in the local HF cache")
def test_corpus_size_and_no_prompt_overlap():
    corpus = C.build_corpus()
    assert corpus.size() <= C.CAP_BYTES
    texts = {}
    for f in ("default.jsonl", "drafting_ext.jsonl"):
        for p in load_prompts(ROOT / "prompts" / f):
            texts[f"{f}:{p['id']}"] = p["text"].encode("utf-8")
    assert len(texts) == 30
    assert C.shared_windows(corpus.data, texts, 32) == []


@pytest.mark.skipif(not HAVE_DATA, reason="data not cached")
def test_the_overlap_check_would_catch_a_leak():
    corpus = C.build_corpus()
    leak = next(r for r in corpus.records[len(corpus.records) // 2:] if len(r) >= 200)[:200]
    hits = C.shared_windows(corpus.data, {"leak": leak}, 32)
    assert hits and all(h[0] == "leak" for h in hits)
