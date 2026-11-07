"""Coverage-aware weighted localization heuristic.

Plain SBFL formulas treat every test equally. In practice a *failing* test
that executes only a handful of statements implicates each of them far more
strongly than a failing test that executes half the program. The
coverage-aware heuristic turns that intuition into test weights.

For test ``t`` covering ``k_t`` elements we set an informativeness weight

    w_t = 1 / (1 + log(k_t))          (k_t > 0, else w_t = 0)

so broad, low-specificity tests are discounted (sub-linearly, to avoid
over-trusting trivially small tests). We then compute an Ochiai-style score
on the *weighted* counts:

    ef_w(e) = sum of w_t over failing tests covering e
    ep_w(e) = sum of w_t over passing tests covering e
    F_w     = sum of w_t over all failing tests
    score(e) = ef_w(e) / sqrt(F_w * (ef_w(e) + ep_w(e)))

This reduces to classic Ochiai when all weights are equal. Unlike the plain
formulas it needs the full coverage matrix, not just the aggregate counts.
"""

from __future__ import annotations

import numpy as np

from faultrank.spectrum import Spectrum

_EPS = 1e-12


def informativeness_weights(spectrum: Spectrum) -> np.ndarray:
    """Informativeness weight per test: ``1 / (1 + log(coverage_size))``."""
    sizes = spectrum.coverage.sum(axis=1).astype(np.float64)
    weights = np.zeros_like(sizes)
    nonzero = sizes > 0
    weights[nonzero] = 1.0 / (1.0 + np.log(sizes[nonzero]))
    return weights


def coverage_aware(spectrum: Spectrum) -> np.ndarray:
    """Coverage-aware weighted-Ochiai suspiciousness per element."""
    weights = informativeness_weights(spectrum)
    fail = spectrum.outcomes
    cov = spectrum.coverage.astype(np.float64)

    wf = weights[fail]
    wp = weights[~fail]
    ef_w = wf @ cov[fail]
    ep_w = wp @ cov[~fail]
    f_w = float(wf.sum())

    den = np.sqrt(f_w * (ef_w + ep_w))
    out = np.zeros_like(ef_w)
    pos = den > _EPS
    out[pos] = ef_w[pos] / den[pos]
    return out
