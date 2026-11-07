from __future__ import annotations

import numpy as np

from faultrank.baselines import failcount, last_frame, random_scores


def test_failcount_equals_ef(toy_spectrum):
    counts = toy_spectrum.counts()
    assert np.array_equal(failcount(counts), counts.ef.astype(float))


def test_random_scores_deterministic(toy_spectrum):
    a = random_scores(toy_spectrum, np.random.default_rng(0))
    b = random_scores(toy_spectrum, np.random.default_rng(0))
    assert np.array_equal(a, b)


def test_random_scores_shape(toy_spectrum):
    scores = random_scores(toy_spectrum, np.random.default_rng(1))
    assert scores.shape == (toy_spectrum.n_elements,)


def test_last_frame_zero_for_unhit():
    # construct a case where elements 1 and 2 are never hit by a failing test
    cov = np.array([[1, 0, 0], [1, 0, 0]], dtype=bool)
    out = np.array([1, 0], dtype=bool)
    from faultrank.spectrum import Spectrum

    sp = Spectrum(coverage=cov, outcomes=out, faults=(0,))
    s = last_frame(sp, sp.counts())
    assert s[1] == 0.0 and s[2] == 0.0
    assert s[0] > 0.0
