from __future__ import annotations

import numpy as np

from faultrank.stats import bootstrap_ci, paired_permutation_test


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 1.0, size=500)
    lo, hi = bootstrap_ci(values, n_boot=1000, seed=0)
    assert lo < values.mean() < hi


def test_bootstrap_ci_empty():
    lo, hi = bootstrap_ci(np.array([]), n_boot=100)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_deterministic():
    values = np.arange(50, dtype=float)
    assert bootstrap_ci(values, seed=1) == bootstrap_ci(values, seed=1)


def test_permutation_detects_difference():
    a = np.ones(60) * 1.0
    b = np.zeros(60)
    p = paired_permutation_test(a, b, n_perm=2000, seed=0)
    assert p < 0.01


def test_permutation_no_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=100)
    p = paired_permutation_test(x, x, n_perm=1000, seed=0)
    assert p == 1.0


def test_permutation_pvalue_never_zero():
    a = np.ones(40) * 3.0
    b = np.zeros(40)
    p = paired_permutation_test(a, b, n_perm=500, seed=0)
    assert p > 0.0


def test_permutation_symmetric_in_sign():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, size=80)
    b = rng.normal(0, 1, size=80)
    p1 = paired_permutation_test(a, b, n_perm=1000, seed=3)
    p2 = paired_permutation_test(b, a, n_perm=1000, seed=3)
    assert abs(p1 - p2) < 1e-9
