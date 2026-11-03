import pytest

from byteprof.stats import (
    bootstrap_ci, paired_bootstrap_diff, paired_relative_change, ratio_of_sums, stratified_counts,
    stratified_sample,
)


def mean(v):
    return sum(v) / len(v)


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    vals = [float(i % 7) for i in range(200)]
    p, lo, hi = bootstrap_ci(vals, mean, seed=3)
    assert p == mean(vals) and lo < p < hi
    assert (p, lo, hi) == bootstrap_ci(vals, mean, seed=3)
    assert (lo, hi) != bootstrap_ci(vals, mean, seed=4)[1:]


def test_bootstrap_ci_width_shrinks_with_n():
    small = bootstrap_ci([0.0, 1.0] * 10, mean, seed=1)
    large = bootstrap_ci([0.0, 1.0] * 500, mean, seed=1)
    assert (large[2] - large[1]) < (small[2] - small[1])


def test_constant_values_give_degenerate_interval():
    assert bootstrap_ci([2.0] * 30, mean) == (2.0, 2.0, 2.0)
    with pytest.raises(ValueError):
        bootstrap_ci([], mean)


def test_paired_diff_uses_pairs_not_marginals():
    a = [1, 0, 1, 1, 0, 1, 0, 1] * 25
    b = list(a)
    d, lo, hi = paired_bootstrap_diff(a, b)
    assert (d, lo, hi) == (0.0, 0.0, 0.0)  # identical pairs: no variance despite a noisy marginal
    a2 = [x + 1 for x in a]
    assert paired_bootstrap_diff(a2, a) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1], [1, 2])


def test_ratio_of_sums_and_relative_change():
    pairs = [(8.0, 4.0), (6.0, 2.0)]
    assert ratio_of_sums(pairs) == pytest.approx(14 / 6)
    worse = [(p[0] * 1.02, p[1]) for p in pairs * 40]
    base = pairs * 40
    ch, lo, hi = paired_relative_change(worse, base)
    assert ch == pytest.approx(0.02) and lo == pytest.approx(0.02) and hi == pytest.approx(0.02)


def test_stratified_counts_and_sample():
    assert stratified_counts(500, 14) == [36] * 10 + [35] * 4
    assert sum(stratified_counts(500, 14)) == 500 and stratified_counts(3, 5) == [1, 1, 1, 0, 0]
    sizes = {f"t{i:02d}": 1000 for i in range(14)}
    s = stratified_sample(sizes, 500, seed=7)
    assert sum(len(v) for v in s.values()) == 500
    assert all(len(set(v)) == len(v) and all(0 <= i < 1000 for i in v) for v in s.values())
    assert s == stratified_sample(sizes, 500, seed=7) and s != stratified_sample(sizes, 500, seed=8)
    # a smaller --limit takes the same per-group prefix sizes pattern
    assert [len(v) for v in stratified_sample(sizes, 20, seed=7).values()] == [2] * 6 + [1] * 8
    with pytest.raises(ValueError):
        stratified_sample({"a": 2}, 5, seed=0)


def test_superset_sample_contains_the_base_sample():
    from byteprof.stats import superset_sample

    sizes = {f"t{i:02d}": 1000 for i in range(14)}
    base = stratified_sample(sizes, 500, seed=7)
    big = superset_sample(sizes, 1500, 500, seed=7)
    assert sum(len(v) for v in big.values()) == 1500
    for t in sizes:
        assert set(base[t]) <= set(big[t]) and len(set(big[t])) == len(big[t])
    assert big == superset_sample(sizes, 1500, 500, seed=7)
    with pytest.raises(ValueError):
        superset_sample(sizes, 100, 500, seed=7)
