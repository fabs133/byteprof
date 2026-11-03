#!/usr/bin/env python3
"""Builds the Zenodo upload bundle in dist/zenodo/ from the git tag (standard library only).

    python tools/build_zenodo_bundle.py [--tag v1.0.0]

Tracked files (paper, report, FINDINGS.md, code snapshot) come from the tag; untracked records
(archive/, results/) come from the working tree. Steps:
  1. privacy check of every text file that goes into a zip: secret-like strings stop the build;
     passwords/e-mail addresses are only counted (the study's model output contains fake ones and
     stays as it is); local home-directory paths are counted;
  2. zips with sorted paths and fixed timestamps; in every text file, absolute Windows home paths
     (C:\\Users\\<name>\\ and C:/Users/<name>/, any case) and the user name become <HOME>;
  3. verification: every file under results/ is in exactly one results zip, and the re-scan of the
     text members of all zips finds no home path.
Binary files (traces, sqlite, .nsys-rep, .npy, pdf) are copied unchanged.
"""
from __future__ import annotations

import argparse
import datetime
import getpass
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "zenodo"
GITHUB = "https://github.com/fabs133/byteprof"
TEXT_EXT = {".json", ".jsonl", ".md", ".txt", ".log", ".toml", ".csv", ".tex", ".bib", ".py", ".sh", ".cff", ".yml", ".yaml", ".ipynb", ".cfg"}
TRACE_SUFFIX = {".sqlite", ".nsys-rep", ".qdrep"}
ZIP_TIME = (1980, 1, 1, 0, 0, 0)

HOME_WIN = re.compile(r"C:[\\/]+Users[\\/]+[^\\/\s\"'<>|:*?]+([\\/]+)", re.I)
HOME_OTHER = re.compile(r"/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+")
SECRET = re.compile(r"\b(hf_[A-Za-z0-9]{20,}|gh[ops]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"
                    r"|\b(?:access_|auth_|bearer_)?token\b\s*[=:]\s*[\"']?[A-Za-z0-9_\-.]{12,}", re.I)
PASSWORD = re.compile(r"password|passwd", re.I)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def git(*args: str, text: bool = False):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=text)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr if text else r.stderr.decode(errors='replace')}")
    return r.stdout


def tag_blob(tag: str, path: str) -> bytes:
    return git("show", f"{tag}:{path}")


def tag_files(tag: str, prefix: str) -> list[str]:
    return sorted(p for p in git("ls-tree", "-r", "--name-only", tag, prefix, text=True).splitlines() if p)


def is_text(name: str) -> bool:
    return Path(name).suffix.lower() in TEXT_EXT


class Redactor:
    def __init__(self) -> None:
        names = {getpass.getuser(), os.environ.get("USERNAME", "")}
        self.names = sorted((n for n in names if len(n) >= 4), key=len, reverse=True)
        self.names_rx = re.compile("|".join(re.escape(n) for n in self.names), re.I) if self.names else None

    def apply(self, data: bytes) -> tuple[bytes, int]:
        s = data.decode("utf-8", "surrogateescape")
        s, n1 = HOME_WIN.subn(lambda m: "<HOME>" + m.group(1), s)
        n2 = 0
        if self.names_rx:
            s, n2 = self.names_rx.subn("<HOME>", s)
        return (s.encode("utf-8", "surrogateescape"), n1 + n2) if n1 + n2 else (data, 0)


def check_text(data: bytes) -> dict:
    s = data.decode("utf-8", "surrogateescape")
    return {"secret": len(SECRET.findall(s)), "password": len(PASSWORD.findall(s)), "email": len(EMAIL.findall(s)),
            "home": len(HOME_WIN.findall(s)) + len(HOME_OTHER.findall(s))}


def write_member(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    zi = zipfile.ZipInfo(arcname, ZIP_TIME)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = 0o644 << 16
    zf.writestr(zi, data, compresslevel=6)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def abstract_text(tex: str) -> str:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        raise SystemExit("no abstract in paper/main.tex")
    a = m.group(1).strip()
    a = re.sub(r"\\x\{\}|\\x\b", "\u00d7", a)               # \newcommand{\x}{\ensuremath{\times}}
    a = re.sub(r"\\emph\{([^}]*)\}", r"\1", a)
    a = a.replace("\\%", "%").replace("~", " ").replace("\\_", "_")
    return re.sub(r"[ \t]*\n[ \t]*", " ", a)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="v1.0.0")
    args = ap.parse_args()
    tag = args.tag
    version = tag.lstrip("v")
    doi_file = ROOT / "zenodo" / "DOI.txt"
    doi = doi_file.read_text(encoding="utf-8").strip() if doi_file.exists() else ""
    if not re.fullmatch(r"10\.\d{4,9}/\S+", doi):
        raise SystemExit("zenodo/DOI.txt is missing or holds no real DOI: stop before Part C")
    git("rev-parse", "--verify", f"{tag}^{{commit}}", text=True)
    red = Redactor()
    log: list[str] = []

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.append(msg)

    # ---- plan: (zip name) -> list of (arcname, source) ; source = ("tag", path) or ("file", Path)
    plan: dict[str, list[tuple[str, tuple]]] = {}
    plan["latex_sources.zip"] = [(p, ("tag", p)) for d in ("paper", "report") for p in tag_files(tag, d + "/")
                                 if Path(p).suffix in (".tex", ".bib")]
    rec: list[tuple[str, tuple]] = [("FINDINGS.md", ("tag", "FINDINGS.md"))]
    for sub in ("archive/tasks", "archive/logs"):
        rec += [(p.relative_to(ROOT).as_posix(), ("file", p)) for p in sorted((ROOT / sub).rglob("*")) if p.is_file()]
    rec.append(("archive/MANIFEST.md", ("file", ROOT / "archive" / "MANIFEST.md")))
    rec += [(p.relative_to(ROOT).as_posix(), ("file", p)) for p in sorted((ROOT / "notes").rglob("*")) if p.is_file()]
    plan["study_records.zip"] = sorted(set(rec))
    core, traces = [], []
    res_files = sorted(p for p in (ROOT / "results").rglob("*") if p.is_file())
    for p in res_files:
        rel = p.relative_to(ROOT).as_posix()
        big = (p.name == "trace.json" or p.suffix.lower() in TRACE_SUFFIX
               or (p.suffix.lower() == ".npy" and "drafting/corpus" in rel))
        (traces if big else core).append((rel, ("file", p)))
    plan["results_core.zip"], plan["results_traces.zip"] = core, traces

    def load(src: tuple) -> bytes:
        return tag_blob(tag, src[1]) if src[0] == "tag" else Path(src[1]).read_bytes()

    # ---- 1. privacy check before anything is written
    tot = {"secret": 0, "password": 0, "email": 0, "home": 0}
    secret_hits: list[str] = []
    home_files: list[str] = []
    nfiles = 0
    for zname, items in plan.items():
        for arc, src in items:
            if not is_text(arc):
                continue
            nfiles += 1
            c = check_text(load(src))
            for k in tot:
                tot[k] += c[k]
            if c["secret"]:
                secret_hits.append(arc)
            if c["home"]:
                home_files.append(arc)
    say(f"privacy check: {nfiles} text files; secret-like strings {tot['secret']}, 'password' lines-matches {tot['password']} "
        f"(model output, kept), e-mail matches {tot['email']} (model output, kept), home-path matches {tot['home']} in {len(home_files)} files")
    if secret_hits:
        say("STOP: secret-like strings in: " + ", ".join(secret_hits))
        return 2

    # ---- 2. write
    if OUT.exists():
        for f in OUT.iterdir():
            f.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    git("archive", "--format=zip", f"--prefix=byteprof-{version}/", "-o", str(OUT / f"byteprof-{version}.zip"), tag)
    (OUT / "byte_tax_paper.pdf").write_bytes(tag_blob(tag, "paper/main.pdf"))
    report_present = bool(tag_files(tag, "report/main.pdf"))
    if report_present:
        (OUT / "byte_tax_technical_report.pdf").write_bytes(tag_blob(tag, "report/main.pdf"))
    else:
        say("report/main.pdf not in the tag: technical report skipped")
    redacted: list[tuple[str, int]] = []
    for zname, items in plan.items():
        with zipfile.ZipFile(OUT / zname, "w") as zf:
            for arc, src in items:
                data = load(src)
                if is_text(arc):
                    data, n = red.apply(data)
                    if n:
                        redacted.append((arc, n))
                write_member(zf, arc, data)
        say(f"wrote {zname}: {len(items)} files, {(OUT / zname).stat().st_size / 1e6:.1f} MB")

    # code snapshot: redaction would change the snapshot, so it must already be clean
    with zipfile.ZipFile(OUT / f"byteprof-{version}.zip") as zf:
        bad = [i.filename for i in zf.infolist() if not i.is_dir() and is_text(i.filename)
               and (HOME_WIN.search(d := zf.read(i).decode("utf-8", "surrogateescape")) or HOME_OTHER.search(d)
                    or (red.names_rx and red.names_rx.search(d)))]
    if bad:
        say("STOP: home path or user name inside the code snapshot (needs a source fix): " + ", ".join(bad))
        return 2

    # ---- 3. verification
    expected = {p.relative_to(ROOT).as_posix() for p in res_files}
    seen: list[str] = []
    for zname in ("results_core.zip", "results_traces.zip"):
        with zipfile.ZipFile(OUT / zname) as zf:
            seen += [i.filename for i in zf.infolist() if not i.is_dir()]
    assert len(seen) == len(set(seen)), "a results file is in both zips"
    assert set(seen) == expected, "results zips do not cover results/ exactly"
    say(f"results: core {len(core)} + traces {len(traces)} = {len(seen)} = files under results/ ({len(expected)})")
    rescan = {"files": 0, "home": 0, "binary_with_user_name": []}
    for zp in sorted(OUT.glob("*.zip")):
        if zp.name == f"byteprof-{version}.zip":
            pass
        with zipfile.ZipFile(zp) as zf:
            for i in zf.infolist():
                if i.is_dir():
                    continue
                data = zf.read(i)
                if is_text(i.filename):
                    rescan["files"] += 1
                    d = data.decode("utf-8", "surrogateescape")
                    rescan["home"] += len(HOME_WIN.findall(d)) + len(HOME_OTHER.findall(d)) + (len(red.names_rx.findall(d)) if red.names_rx else 0)
                elif red.names and any(n.encode().lower() in data.lower() or n.encode("utf-16le").lower() in data.lower() for n in red.names):
                    rescan["binary_with_user_name"].append(f"{zp.name}:{i.filename}")
    say(f"re-scan of text members in the zips: {rescan['files']} files, home-path/user-name hits {rescan['home']}")
    if rescan["binary_with_user_name"]:
        say("binary members containing the user name (not altered): " + ", ".join(rescan["binary_with_user_name"]))
    if rescan["home"]:
        say("STOP: home paths remain after redaction")
        return 2

    # ---- 4. README_ZENODO.md, then sums, then the form
    bin_note = ("Binary files (PDFs, `.nsys-rep`, `.sqlite`, `.npy`) were copied unchanged. "
                + (f"{len(rescan['binary_with_user_name'])} of them contain the user name in binary form: "
                   + ", ".join(f"`{x}`" for x in rescan["binary_with_user_name"]) + "."
                   if rescan["binary_with_user_name"] else "A byte search found no user name in them."))
    red_lines = "\n".join(f"- `{a}` ({n} replacement{'s' if n != 1 else ''})" for a, n in sorted(redacted)) or "- none"
    readme = f"""# The Byte Tax: Decode Speed of Byte-Level Language Models on a Consumer GPU

Zenodo record, version {version}. DOI: [{doi}](https://doi.org/{doi}). Code repository: {GITHUB}.

One record holds the paper, the technical report, a code snapshot, the raw results and the study records of the
Bolmo 1B vs OLMo 2 1B decode-speed study (13 hypothesis-driven studies; hypotheses were written down before each
measurement). All numbers come from one RTX 3060 Ti (8 GB) on Windows at batch size 1, with greedy decoding and a fixed seed.

## Licenses

- Code (`byteprof-{version}.zip`): Apache License 2.0.
- Everything else in this record (paper, technical report, LaTeX sources, study records, results): Creative Commons Attribution 4.0 International (CC BY 4.0).

## Files

### `byte_tax_paper.pdf`
The preprint, compiled from `paper/main.tex` at tag `{tag}`.

### `byte_tax_technical_report.pdf`
{"The companion technical report: it lists all 67 hypotheses with verdicts and documents every study, including the negative results." if report_present else "Not included in this build."}

### `{f'byteprof-{version}.zip'}`
Snapshot of the repository at tag `{tag}` (`git archive`): the `byteprof` package with the optimized decode loops, drafting and quantization code, the Triton/CUDA kernels, configs, prompts, tests, study scripts (`scripts/`), notes and the findings log. Raw results are not part of it.

### `latex_sources.zip`
The `.tex` and `.bib` sources of the paper (`paper/`) and the technical report (`report/`).

### `study_records.zip`
`FINDINGS.md` (the full findings log, with the pre-registered hypotheses and their verdicts), `archive/tasks/` (the written task specifications of each study), `archive/logs/` (console logs of early measurement runs), `archive/MANIFEST.md` (record of the repository tidy-up) and `notes/` (design notes on the Bolmo decode loop).

### `results_core.zip`
Raw results from `results/` without the large profiler files: per-run reports and JSON, logs, recordings and evaluation outputs ({len(core)} files).

### `results_traces.zip`
The large profiler files from `results/`: `trace.json`, `*.sqlite`, `*.nsys-rep`, `*.qdrep` and the drafting corpus `.npy` files ({len(traces)} files). Together with `results_core.zip` it contains every file under `results/` exactly once ({len(core) + len(traces)} files).

### `SHA256SUMS.txt`
SHA-256 of every other file in this record.

## Reproduce

Clone or unpack the code snapshot and follow its `README.md` (setup, tests, a single profile run); `scripts/README.md` lists which script ran which study and which section of `FINDINGS.md` it feeds. Unpack `results_core.zip` and `results_traces.zip` into the repository root to get the `results/` folder the reports refer to.

## Redactions

Local home-directory paths replaced with `<HOME>`; no other changes. The password and e-mail strings that occur in the results are fake values from the model's generated text and were kept. {bin_note}

Text files with replacements:

{red_lines}
"""
    (OUT / "README_ZENODO.md").write_bytes(readme.encode("utf-8"))
    sums = [f for f in sorted(OUT.iterdir()) if f.is_file() and f.name != "SHA256SUMS.txt"]
    (OUT / "SHA256SUMS.txt").write_text("".join(f"{sha256(f)}  {f.name}\n" for f in sums), encoding="utf-8", newline="\n")
    names = [f.name for f in sorted(OUT.iterdir())]
    form = f"""# Zenodo form (not uploaded; copy into the form)

- **Resource type:** Publication / Preprint
- **Title:** The Byte Tax: Decode Speed of Byte-Level Language Models on a Consumer GPU
- **Creator:** Rempel, Fabio-Eric; ORCID 0009-0008-8447-6159; affiliation: Independent researcher
- **Version:** {version}
- **Publication date:** {datetime.date.today().isoformat()}
- **Language:** English
- **Licenses:** Creative Commons Attribution 4.0 International; Apache License 2.0
- **Keywords:** byte-level language models; tokenizer-free models; speculative decoding; quantization; LLM inference efficiency; CUDA graphs; Bolmo; OLMo 2; pre-registration
- **Related works:**
  - "Is supplemented by" -> {GITHUB} (resource type: software)
  - "References" -> 10.1038/s41586-026-11111-4 (Bolmo, Nature 2026)

## Description

(Abstract of `paper/main.tex`; the LaTeX macros `\\x{{}}`, `\\emph{{}}`, `\\%` and `~` are written out as x-sign, plain text, % and space.)

{abstract_text(tag_blob(tag, 'paper/main.tex').decode('utf-8'))}

Files in this record:

- `byte_tax_paper.pdf`: the preprint.
- `byte_tax_technical_report.pdf`: the technical report with all 67 hypotheses, verdicts and every study.
- `byteprof-{version}.zip`: code snapshot (Apache-2.0).
- `latex_sources.zip`: LaTeX sources of paper and report.
- `study_records.zip`: findings log, task specifications, logs, notes.
- `results_core.zip`: raw results without the large profiler files.
- `results_traces.zip`: large profiler files (traces, sqlite, nsys-rep, drafting corpus arrays).
- `README_ZENODO.md`: description of the files, licenses, redactions.
- `SHA256SUMS.txt`: checksums of the other files.
"""
    (OUT / "zenodo_form.md").write_bytes(form.encode("utf-8"))
    # README and form were written after the zip scan: scan them too
    for nm in ("README_ZENODO.md", "zenodo_form.md"):
        c = check_text((OUT / nm).read_bytes())
        assert c["home"] == 0 and c["secret"] == 0, nm
    # verify sums
    for line in (OUT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, n = line.split("  ", 1)
        assert sha256(OUT / n) == h, n
    for f in sorted(OUT.iterdir()):
        say(f"{f.stat().st_size:>13,d}  {f.name}")
    if max(f.stat().st_size for f in OUT.iterdir()) > 2 * 1024 ** 3:
        say("STOP: a file is larger than 2 GB")
        return 2
    say("redacted files: " + json.dumps([a for a, _ in sorted(redacted)]))
    (ROOT / "dist" / "build_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
