from __future__ import annotations

import numpy as np

from faultrank.formulas import ochiai
from faultrank.heuristic import coverage_aware, informativeness_weights
from faultrank.spectrum import Spectrum


def test_coverage_aware_finds_fault(toy_spectrum):
    scores = coverage_aware(toy_spectrum)
    assert int(np.argmax(scores)) == 1


def test_weights_discount_large_tests():
    cov = np.array([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=bool)
    out = np.array([1, 1], dtype=bool)
    sp = Spectrum(coverage=cov, outcomes=out, faults=(0,))
    w = informativeness_weights(sp)
    # the broad test (4 elements) should weigh less than the narrow one (1)
    assert w[0] < w[1]


def test_equal_size_tests_reduce_to_ochiai():
    # when every test covers exactly the same number of elements, the
    # coverage-aware weights are all equal and the score matches Ochiai.
    cov = np.array(
        [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]],
        dtype=bool,
    )
    assert np.all(cov.sum(axis=1) == 2)
    out = np.array([1, 1, 0, 0], dtype=bool)
    sp = Spectrum(coverage=cov, outcomes=out, faults=(0,))
    ca = coverage_aware(sp)
    oc = ochiai(sp.counts())
    assert np.allclose(ca, oc)


def test_coverage_aware_non_negative(toy_spectrum):
    scores = coverage_aware(toy_spectrum)
    assert np.all(scores >= 0.0)


def test_empty_test_gets_zero_weight():
    cov = np.array([[0, 0], [1, 1]], dtype=bool)
    out = np.array([0, 1], dtype=bool)
    sp = Spectrum(coverage=cov, outcomes=out, faults=(0,))
    w = informativeness_weights(sp)
    assert w[0] == 0.0
