"""Loads the ground truth recorded by tools/draft_truth.py as atoms (byte | flag << 8)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import make_atom

OFFSET, BOUNDARY = 4, 260  # plain id = byte + 4; boundary-flagged id = plain + 260
DEFAULT_TYPES = {"prose_en": "prose_en", "code_py": "code_py", "prose_de": "prose_de", "json_struct": "json",
                 "unicode_mix": "unicode_mix"}
PREFIX_TYPES = {"en": "prose_en", "py": "code_py", "de": "prose_de", "json": "json", "uni": "unicode_mix"}


def raw_to_atom(raw: int) -> int | None:
    """Raw model id -> atom, or None for special tokens (eos, pad, <bpe_token_end>)."""
    flag = raw >= BOUNDARY
    plain = raw - BOUNDARY if flag else raw
    if not OFFSET <= plain < OFFSET + 256:
        return None
    return make_atom(plain - OFFSET, flag)


def atom_to_raw(atom: int) -> int:
    return (atom & 255) + OFFSET + (BOUNDARY if atom >> 8 else 0)


@dataclass
class Case:
    id: str
    set: str
    type: str
    text: str
    ctx_atoms: list[int]  # prompt bytes (all but the last) with their prefill boundary flags
    truth: dict[int, list[int]] = field(default_factory=dict)  # new bytes n -> atoms a_0 .. a_n
    raw: dict[int, list[int]] = field(default_factory=dict)

    @property
    def prompt_bytes(self) -> bytes:
        return bytes(a & 255 for a in self.ctx_atoms)


def load_truth(path: str | Path) -> dict[str, Case]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: dict[str, Case] = {}
    for pid, rec in obj["prompts"].items():
        typ = DEFAULT_TYPES[pid] if rec["set"] == "default" else PREFIX_TYPES[pid.split("_")[0]]
        ctx = []
        for tid, m in zip(rec["ids"][:-1], rec["mask"]):  # mask is aligned with ids[:-1]
            a = raw_to_atom(tid)
            if a is not None:
                ctx.append(make_atom(a & 255, m))
        c = Case(pid, rec["set"], typ, rec["text"], ctx)
        for n, raws in rec["raw"].items():
            atoms = []
            for r in raws:
                a = raw_to_atom(r)
                if a is None:
                    break
                atoms.append(a)
            c.truth[int(n)] = atoms
            c.raw[int(n)] = raws
        cases[pid] = c
    return cases


def repetition_rate(prompt: bytes, output: bytes, w: int = 8) -> tuple[int, int]:
    """(repeated windows, windows): w-byte windows that end in the output and occurred earlier in prompt + output."""
    data = prompt + output
    seen: set[bytes] = set()
    rep = tot = 0
    for s in range(len(data) - w + 1):
        win = data[s:s + w]
        if s + w > len(prompt):  # window reaches into the output
            tot += 1
            rep += win in seen
        seen.add(win)
    return rep, tot
