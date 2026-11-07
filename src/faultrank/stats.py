"""Honest uncertainty: bootstrap CIs and a paired permutation test.

These are deliberately simple, dependency-free resampling procedures so the
reported numbers come with real error bars rather than bare point estimates.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return (lo, hi)


def paired_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided paired permutation (sign-flip) p-value for mean(a) != mean(b).

    Tests the null that the paired differences ``a - b`` are symmetric about
    zero by randomly flipping the sign of each difference.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    diff = a - b
    n = diff.size
    if n == 0:
        return float("nan")
    observed = abs(float(diff.mean()))
    if np.allclose(diff, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.integers(0, 2, size=n) * 2 - 1
        if abs(float((diff * signs).mean())) >= observed - 1e-15:
            count += 1
    # add-one smoothing so a p-value is never exactly 0
    return (count + 1) / (n_perm + 1)
