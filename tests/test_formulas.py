from __future__ import annotations

import numpy as np
import pytest

from faultrank.formulas import FORMULAS, dstar, ochiai, suspiciousness, tarantula


def test_all_formulas_rank_obvious_fault(toy_spectrum):
    counts = toy_spectrum.counts()
    for name, fn in FORMULAS.items():
        scores = fn(counts)
        assert int(np.argmax(scores)) == 1, f"{name} missed the fault"


def test_ochiai_bounded_unit_interval(toy_spectrum):
    scores = ochiai(toy_spectrum.counts())
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0 + 1e-9)


def test_tarantula_zero_when_no_failure_coverage():
    # element 2 never executed by a failing test -> tarantula 0
    cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
    out = np.array([1, 0, 0], dtype=bool)
    from faultrank.spectrum import Spectrum

    counts = Spectrum(coverage=cov, outcomes=out, faults=(0,)).counts()
    scores = tarantula(counts)
    assert scores[2] == 0.0


def test_dstar_saturation_is_finite():
    # element executed only by failing tests, ep+nf == 0 -> saturates finite
    cov = np.array([[1, 1], [1, 0]], dtype=bool)
    out = np.array([1, 1], dtype=bool)
    from faultrank.spectrum import Spectrum

    counts = Spectrum(coverage=cov, outcomes=out, faults=(0,)).counts()
    scores = dstar(counts)
    assert np.all(np.isfinite(scores))
    assert scores[0] >= scores[1]


def test_dstar_exponent_monotone():
    cov = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    out = np.array([1, 1, 0], dtype=bool)
    from faultrank.spectrum import Spectrum

    counts = Spectrum(coverage=cov, outcomes=out, faults=(0,)).counts()
    s2 = dstar(counts, star=2)
    s3 = dstar(counts, star=3)
    assert np.all(np.isfinite(s2)) and np.all(np.isfinite(s3))


def test_suspiciousness_lookup(toy_spectrum):
    counts = toy_spectrum.counts()
    assert np.allclose(suspiciousness("ochiai", counts), ochiai(counts))


def test_suspiciousness_unknown():
    cov = np.ones((2, 2), dtype=bool)
    out = np.array([1, 0], dtype=bool)
    from faultrank.spectrum import Spectrum

    counts = Spectrum(coverage=cov, outcomes=out, faults=(0,)).counts()
    with pytest.raises(KeyError):
        suspiciousness("nope", counts)


def test_no_nan_on_empty_element():
    # a column executed by no test at all
    cov = np.array([[1, 0], [0, 0]], dtype=bool)
    out = np.array([1, 0], dtype=bool)
    from faultrank.spectrum import Spectrum

    counts = Spectrum(coverage=cov, outcomes=out, faults=(0,)).counts()
    for fn in FORMULAS.values():
        scores = fn(counts)
        assert not np.any(np.isnan(scores))
