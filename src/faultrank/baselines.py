"""Baseline localizers to benchmark the SBFL methods against.

* ``random`` -- seeded uniform-random suspiciousness. The floor any useful
  method must clear.
* ``failcount`` -- rank by ``ef`` alone (how many failing tests cover the
  element). The naive "blame whatever the failing tests touch most" rule;
  it ignores passing tests entirely.
* ``last`` -- a "last-frame" style baseline: among elements covered by at
  least one failing test, prefer those appearing *latest* in the program
  (highest index), mimicking blaming the most recently executed statement.
  Elements never hit by a failing test score 0.
"""

from __future__ import annotations

import numpy as np

from faultrank.spectrum import Spectrum, SpectrumCounts


def random_scores(spectrum: Spectrum, rng: np.random.Generator) -> np.ndarray:
    return rng.random(spectrum.n_elements)


def failcount(counts: SpectrumCounts) -> np.ndarray:
    return counts.ef.astype(np.float64)


def last_frame(spectrum: Spectrum, counts: SpectrumCounts) -> np.ndarray:
    """Prefer later elements that at least one failing test executed."""
    n = spectrum.n_elements
    hit = counts.ef > 0
    position = np.arange(1, n + 1, dtype=np.float64) / n
    return np.where(hit, position, 0.0)
