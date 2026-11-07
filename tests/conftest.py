"""Shared fixtures and strategies for the test suite."""

from __future__ import annotations

import numpy as np
import pytest

from faultrank.spectrum import Spectrum


@pytest.fixture
def toy_spectrum() -> Spectrum:
    """A tiny hand-built spectrum where element 1 is the obvious fault.

    Four tests, three elements. Element 1 is executed by both failing tests
    and no passing test -- so it must be ranked most suspicious.
    """
    coverage = np.array(
        [
            [1, 1, 0],  # fail
            [0, 1, 1],  # fail
            [1, 0, 1],  # pass
            [1, 0, 0],  # pass
        ],
        dtype=bool,
    )
    outcomes = np.array([1, 1, 0, 0], dtype=bool)
    return Spectrum(coverage=coverage, outcomes=outcomes, faults=(1,))
