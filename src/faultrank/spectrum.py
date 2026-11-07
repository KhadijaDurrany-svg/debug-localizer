"""Coverage spectrum data model.

A :class:`Spectrum` holds, for a single program version under test:

* a boolean coverage matrix of shape ``(n_tests, n_elements)`` where
  ``coverage[t, e]`` is ``True`` if test ``t`` executed element ``e``;
* a boolean outcome vector of length ``n_tests`` where ``True`` means the
  test *failed*;
* the ground-truth faulty element indices (used only for evaluation, never
  by a localizer).

From the matrix we derive the four classic SBFL counts per element:

* ``ef`` executed by failing tests,
* ``ep`` executed by passing tests,
* ``nf`` not executed by failing tests,
* ``np_`` not executed by passing tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectrumCounts:
    """Per-element SBFL counts, each a 1-D int array of length ``n_elements``."""

    ef: np.ndarray
    ep: np.ndarray
    nf: np.ndarray
    np_: np.ndarray
    total_fail: int
    total_pass: int

    def __post_init__(self) -> None:
        n = self.ef.shape[0]
        for name in ("ep", "nf", "np_"):
            if getattr(self, name).shape[0] != n:
                raise ValueError("count arrays must share a length")


@dataclass(frozen=True)
class Spectrum:
    """A coverage spectrum for one faulty program version."""

    coverage: np.ndarray  # bool, shape (n_tests, n_elements)
    outcomes: np.ndarray  # bool, shape (n_tests,); True == failing test
    faults: tuple[int, ...]  # ground-truth faulty element indices
    element_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.coverage.ndim != 2:
            raise ValueError("coverage must be a 2-D matrix")
        if self.coverage.dtype != np.bool_:
            object.__setattr__(self, "coverage", self.coverage.astype(bool))
        if self.outcomes.dtype != np.bool_:
            object.__setattr__(self, "outcomes", self.outcomes.astype(bool))
        if self.outcomes.shape[0] != self.coverage.shape[0]:
            raise ValueError("outcomes length must equal number of tests")
        if not self.faults:
            raise ValueError("at least one faulty element is required")
        if any(not (0 <= f < self.n_elements) for f in self.faults):
            raise ValueError("fault index out of range")

    @property
    def n_tests(self) -> int:
        return int(self.coverage.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.coverage.shape[1])

    def counts(self) -> SpectrumCounts:
        """Compute the four SBFL counts per element."""
        fail = self.outcomes
        cov = self.coverage
        total_fail = int(fail.sum())
        total_pass = int((~fail).sum())
        ef = cov[fail].sum(axis=0).astype(np.int64)
        ep = cov[~fail].sum(axis=0).astype(np.int64)
        nf = total_fail - ef
        np_ = total_pass - ep
        return SpectrumCounts(
            ef=ef, ep=ep, nf=nf, np_=np_, total_fail=total_fail, total_pass=total_pass
        )

    def checksum(self) -> str:
        """Stable SHA-256 over the coverage, outcomes and fault indices."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.coverage).tobytes())
        h.update(np.ascontiguousarray(self.outcomes).tobytes())
        h.update(np.asarray(self.faults, dtype=np.int64).tobytes())
        return h.hexdigest()
