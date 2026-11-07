from __future__ import annotations

import numpy as np
import pytest

from faultrank.spectrum import Spectrum


def test_counts_basic(toy_spectrum):
    c = toy_spectrum.counts()
    # element 1 executed by both failing tests, no passing test
    assert c.ef[1] == 2
    assert c.ep[1] == 0
    assert c.nf[1] == 0
    assert c.total_fail == 2
    assert c.total_pass == 2


def test_counts_sum_invariant(toy_spectrum):
    c = toy_spectrum.counts()
    assert np.all(c.ef + c.nf == c.total_fail)
    assert np.all(c.ep + c.np_ == c.total_pass)


def test_dimensions(toy_spectrum):
    assert toy_spectrum.n_tests == 4
    assert toy_spectrum.n_elements == 3


def test_checksum_stable(toy_spectrum):
    assert toy_spectrum.checksum() == toy_spectrum.checksum()


def test_checksum_changes_with_data(toy_spectrum):
    cov = toy_spectrum.coverage.copy()
    cov[0, 2] = ~cov[0, 2]
    other = Spectrum(coverage=cov, outcomes=toy_spectrum.outcomes, faults=(1,))
    assert other.checksum() != toy_spectrum.checksum()


def test_requires_fault():
    cov = np.ones((2, 2), dtype=bool)
    out = np.array([1, 0], dtype=bool)
    with pytest.raises(ValueError):
        Spectrum(coverage=cov, outcomes=out, faults=())


def test_fault_out_of_range():
    cov = np.ones((2, 2), dtype=bool)
    out = np.array([1, 0], dtype=bool)
    with pytest.raises(ValueError):
        Spectrum(coverage=cov, outcomes=out, faults=(5,))


def test_outcome_length_mismatch():
    cov = np.ones((2, 2), dtype=bool)
    out = np.array([1, 0, 1], dtype=bool)
    with pytest.raises(ValueError):
        Spectrum(coverage=cov, outcomes=out, faults=(0,))


def test_coverage_must_be_2d():
    with pytest.raises(ValueError):
        Spectrum(
            coverage=np.array([True, False]),
            outcomes=np.array([True]),
            faults=(0,),
        )
