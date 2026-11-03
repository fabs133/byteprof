"""Pure statistics helpers for the quality evaluation (no torch)."""

from __future__ import annotations

import random
from collections.abc import Sequence


def bootstrap_ci(values: Sequence[float], stat, *, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """(point estimate, lo, hi): percentile bootstrap over items; `stat(list)` -> float."""
    n = len(values)
    if n == 0:
        raise ValueError("no values")
    rng = random.Random(seed)
    point = stat(list(values))
    stats = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return point, lo, hi


def ratio_of_sums(pairs: Sequence[tuple[float, float]]) -> float:
    """sum(numerator) / sum(denominator) over (num, den) pairs, e.g. total bits / total bytes."""
    return sum(p[0] for p in pairs) / sum(p[1] for p in pairs)


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], *, n_boot: int = 2000, seed: int = 0,
                          alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean(a - b) over paired items with a percentile bootstrap CI (same resampled indices for both)."""
    if len(a) != len(b):
        raise ValueError(f"paired samples need equal length, got {len(a)} and {len(b)}")
    diffs = [x - y for x, y in zip(a, b)]
    return bootstrap_ci(diffs, lambda v: sum(v) / len(v), n_boot=n_boot, seed=seed, alpha=alpha)


def paired_relative_change(pairs_a: Sequence[tuple[float, float]], pairs_b: Sequence[tuple[float, float]], *,
                           n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Relative change ratio_of_sums(a) / ratio_of_sums(b) - 1 with a paired bootstrap over chunks."""
    if len(pairs_a) != len(pairs_b):
        raise ValueError("paired samples need equal length")
    idx = list(range(len(pairs_a)))

    def stat(sample: list[int]) -> float:
        return ratio_of_sums([pairs_a[i] for i in sample]) / ratio_of_sums([pairs_b[i] for i in sample]) - 1.0

    return bootstrap_ci(idx, stat, n_boot=n_boot, seed=seed, alpha=alpha)


def stratified_counts(total: int, n_groups: int) -> list[int]:
    """Split `total` items over `n_groups` as evenly as possible; the first groups get the remainder."""
    base, extra = divmod(total, n_groups)
    return [base + (1 if i < extra else 0) for i in range(n_groups)]


def stratified_sample(sizes: dict[str, int], total: int, seed: int) -> dict[str, list[int]]:
    """Seeded sample without replacement: {group: sorted indices}; groups in sorted-name order."""
    names = sorted(sizes)
    counts = stratified_counts(total, len(names))
    rng = random.Random(seed)
    out: dict[str, list[int]] = {}
    for name, k in zip(names, counts):
        if k > sizes[name]:
            raise ValueError(f"group {name!r} has {sizes[name]} items, {k} requested")
        out[name] = sorted(rng.sample(range(sizes[name]), k))
    return out


def superset_sample(sizes: dict[str, int], total: int, base_total: int, seed: int) -> dict[str, list[int]]:
    """Like stratified_sample(sizes, total, seed), but guaranteed to contain stratified_sample(sizes,
    base_total, seed): the base items are kept and the rest of each group is drawn from the remainder."""
    base = stratified_sample(sizes, base_total, seed)
    names = sorted(sizes)
    counts = stratified_counts(total, len(names))
    rng = random.Random(seed + 1)
    out: dict[str, list[int]] = {}
    for name, k in zip(names, counts):
        if k > sizes[name]:
            raise ValueError(f"group {name!r} has {sizes[name]} items, {k} requested")
        if k < len(base[name]):
            raise ValueError(f"total={total} is smaller than base_total={base_total} for group {name!r}")
        rest = [i for i in range(sizes[name]) if i not in set(base[name])]
        out[name] = sorted(base[name] + rng.sample(rest, k - len(base[name])))
    return out
