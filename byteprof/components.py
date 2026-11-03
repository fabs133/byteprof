"""Map dotted module names to logical components (pure, no torch)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentMatch:
    component: str
    module_name: str


def _is_descendant(name: str, ancestor: str) -> bool:
    return ancestor == "" or name.startswith(ancestor + ".")


def match_components(
    module_names: Iterable[str], patterns: dict[str, list[str]]
) -> list[ComponentMatch]:
    """Assign modules to components.

    Rules:
      * a module belongs to the first component (config order) with a matching regex;
      * a module nested inside an already-selected module is dropped, so time is
        never counted twice;
      * the root module ("") is never matched (it is the step marker).
    """
    compiled = [(comp, [re.compile(p) for p in pats]) for comp, pats in patterns.items()]
    # Parents before children: shallower names first, then stable by name.
    ordered = sorted((n for n in module_names if n), key=lambda n: (n.count("."), n))
    selected: list[ComponentMatch] = []
    for name in ordered:
        if any(_is_descendant(name, m.module_name) for m in selected):
            continue
        for comp, regexes in compiled:
            if any(r.search(name) for r in regexes):
                selected.append(ComponentMatch(comp, name))
                break
    return selected


def unmatched_components(
    matches: Iterable[ComponentMatch], patterns: dict[str, list[str]]
) -> list[str]:
    found = {m.component for m in matches}
    return [c for c in patterns if c not in found]
