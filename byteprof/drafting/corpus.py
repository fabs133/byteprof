"""The fixed n-gram corpus (<= 20 MB) and the check that no prompt text is in it.

Sources (pinned): WikiText-2 raw **train** (`Salesforce/wikitext`), a deterministic subset of the
Python standard library sources of the local install (top-level modules and packages, without test
suites), and a German sample (German side of `Helsinki-NLP/opus_books` de-en, every third row).
Prompts are never taken from these (`shared_windows` checks it: no shared 32-byte window).
"""

from __future__ import annotations

import hashlib
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WIKITEXT_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_TRAIN = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
OPUS_REV = "1f9f6191d0e91a3c539c2595e2fe48fc1420de9b"
OPUS_FILE = "de-en/train-00000-of-00001.parquet"
CAP_BYTES = 20_000_000
GERMAN_STRIDE = 3  # every third opus_books row
STDLIB_SKIP_DIRS = {"test", "tests", "idlelib", "site-packages", "turtledemo", "lib2to3", "ensurepip",
                    "__pycache__", "tkinter", "pydoc_data", "distutils"}
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


@dataclass
class Corpus:
    records: list[bytes]
    sources: list[dict] = field(default_factory=list)

    @property
    def data(self) -> bytes:
        return b"".join(self.records)

    def size(self) -> int:
        return sum(len(r) for r in self.records)


def _hf_file(repo: str, rev: str, filename: str) -> Path:
    p = HF_CACHE / ("datasets--" + repo.replace("/", "--")) / "snapshots" / rev / filename
    if not p.is_file():
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=rev))
    return p


def stdlib_files() -> list[Path]:
    root = Path(sysconfig.get_paths()["stdlib"])
    out = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if any(part in STDLIB_SKIP_DIRS for part in rel.parts[:-1]):
            continue
        out.append(p)
    return out


def _keep(rel: str, frac: float) -> bool:
    return int.from_bytes(hashlib.md5(rel.encode()).digest()[:4], "little") / 2**32 < frac


def build_corpus(cap: int = CAP_BYTES) -> Corpus:
    import pyarrow.parquet as pq

    records: list[bytes] = []
    sources: list[dict] = []

    wt = pq.read_table(_hf_file("Salesforce/wikitext", WIKITEXT_REV, WIKITEXT_TRAIN)).column("text").to_pylist()
    wrec = [t.encode("utf-8") for t in wt if t]
    sources.append({"name": "wikitext-2-raw-v1 train", "repo": "Salesforce/wikitext", "revision": WIKITEXT_REV,
                    "file": WIKITEXT_TRAIN, "bytes": sum(map(len, wrec)), "records": len(wrec)})
    records += wrec

    de = pq.read_table(_hf_file("Helsinki-NLP/opus_books", OPUS_REV, OPUS_FILE)).column("translation").to_pylist()
    grec = [(r["de"] + "\n").encode("utf-8") for i, r in enumerate(de) if i % GERMAN_STRIDE == 0 and r["de"]]
    sources.append({"name": "opus_books de-en, German side, every 3rd row", "repo": "Helsinki-NLP/opus_books",
                    "revision": OPUS_REV, "file": OPUS_FILE, "bytes": sum(map(len, grec)), "records": len(grec)})
    records += grec

    used = sum(map(len, records))
    budget = cap - used
    files = stdlib_files()
    sizes = {p: p.stat().st_size for p in files}
    total = sum(sizes.values())
    frac = min(1.0, budget / total) if total else 0.0
    root = Path(sysconfig.get_paths()["stdlib"])
    prec: list[bytes] = []
    nbytes = 0
    for p in files:
        if not _keep(str(p.relative_to(root)).replace("\\", "/"), frac):
            continue
        b = p.read_bytes().decode("utf-8", errors="replace").replace("\r\n", "\n").encode("utf-8")
        if nbytes + len(b) > budget:
            continue
        prec.append(b)
        nbytes += len(b)
    sources.append({"name": "Python standard library sources (local install)", "path": str(root),
                    "python": sysconfig.get_python_version(), "bytes": nbytes, "records": len(prec),
                    "candidate_files": len(files), "candidate_bytes": total, "kept_fraction": round(frac, 4),
                    "skipped_dirs": sorted(STDLIB_SKIP_DIRS)})
    records += prec
    c = Corpus(records, sources)
    assert c.size() <= cap, (c.size(), cap)
    return c


def window_hashes(data: np.ndarray, w: int) -> np.ndarray:
    """64-bit polynomial hash of every w-byte window of `data` (uint8 array), length len-w+1."""
    n = len(data) - w + 1
    if n <= 0:
        return np.zeros(0, np.uint64)
    P = np.uint64(1099511628211)
    h = np.zeros(n, np.uint64)
    with np.errstate(over="ignore"):
        for t in range(w):
            h = h * P + (data[t:t + n].astype(np.uint64) + np.uint64(1))
    return h


def shared_windows(corpus: bytes, texts: dict[str, bytes], w: int = 32) -> list[tuple[str, int, int]]:
    """All (text name, offset in text, offset in corpus) with an identical w-byte window (exact check)."""
    arr = np.frombuffer(corpus, np.uint8)
    ch = window_hashes(arr, w)
    out: list[tuple[str, int, int]] = []
    ph: dict[int, list[tuple[str, int]]] = {}
    for name, t in texts.items():
        for off, hv in enumerate(window_hashes(np.frombuffer(t, np.uint8), w).tolist()):
            ph.setdefault(hv, []).append((name, off))
    if not ph:
        return out
    keys = np.fromiter(ph.keys(), np.uint64, len(ph))
    hits = np.flatnonzero(np.isin(ch, keys))
    for pos in hits.tolist():
        win = corpus[pos:pos + w]
        for name, off in ph.get(int(ch[pos]), []):
            if texts[name][off:off + w] == win:
                out.append((name, off, pos))
    return out


def segment_with_tokenizer(corpus: Corpus, tokenizer_json: Path, chunk_bytes: int = 1_000_000):
    """Subword segmentation of the corpus with the dolma2 tokenizer (the one OLMo 2 uses and Bolmo's
    patches follow). Returns (flags uint8[len(data)] = 1 on the last byte of each token,
    token ids uint32, token end byte positions int64 in `data`)."""
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(tokenizer_json))
    flags = np.zeros(corpus.size(), np.uint8)
    ids: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    base = 0
    group: list[bytes] = []
    size = 0

    def flush():
        nonlocal base, group, size
        if not group:
            return
        raw = b"".join(group)
        text = raw.decode("utf-8")
        enc = tk.encode(text, add_special_tokens=False)
        buf = np.frombuffer(raw, np.uint8)
        starts = np.flatnonzero((buf & 0xC0) != 0x80)  # char index -> byte offset
        char_end = np.fromiter((o[1] for o in enc.offsets), np.int64, len(enc.offsets))
        byte_end = np.where(char_end < len(starts), starts[np.minimum(char_end, len(starts) - 1)], len(raw))
        last = byte_end - 1
        flags[base + last[last >= 0]] = 1
        ids.append(np.asarray(enc.ids, np.uint32))
        ends.append(base + byte_end)
        base += len(raw)
        group, size = [], 0

    for r in corpus.records:
        group.append(r)
        size += len(r)
        if size >= chunk_bytes:
            flush()
    flush()
    return flags, np.concatenate(ids), np.concatenate(ends)
