"""Patch-level drafting for Bolmo: pure-Python (CPU) drafters and the round-rule simulator.

An *atom* is Bolmo's fused (byte, boundary flag) symbol, stored as `byte | flag << 8`. The encoder's
input is the byte only; the flag says that the byte ends a patch. See TASK_drafting.md for the round
rule that `sim.simulate` implements and the real loop must follow.
"""

from __future__ import annotations

FLAG_SHIFT = 8


def make_atom(byte: int, flag: int | bool) -> int:
    return byte | (int(bool(flag)) << FLAG_SHIFT)


def byte_of(atom: int) -> int:
    return atom & 0xFF


def flag_of(atom: int) -> int:
    return atom >> FLAG_SHIFT
