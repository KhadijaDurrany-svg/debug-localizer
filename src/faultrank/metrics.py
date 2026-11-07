"""Localization-accuracy metrics, computed per program and aggregated.

For a single program and a vector of per-element suspiciousness scores we
derive the fault's mid-rank and from it:

* ``hit@1`` / ``hit@5`` -- 1.0 if the fault's mid-rank is <= 1 / <= 5.
* ``reciprocal_rank`` -- ``1 / rank`` (averaged this gives MRR).
* ``wasted_effort`` -- ``rank - 1`` elements inspected before the fault.
* ``exam`` -- ``rank / n_elements``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faultrank.ranking import exam_score, fault_rank


@dataclass(frozen=True)
class ProgramScore:
    rank: float
    reciprocal_rank: float
    wasted_effort: float
    exam: float
    hit1: float
    hit5: float


def score_program(
    scores: np.ndarray, faults: tuple[int, ...], n_elements: int
) -> ProgramScore:
    rank = fault_rank(scores, faults)
    return ProgramScore(
        rank=rank,
        reciprocal_rank=1.0 / rank,
        wasted_effort=rank - 1.0,
        exam=exam_score(scores, faults, n_elements),
        hit1=1.0 if rank <= 1.0 else 0.0,
        hit5=1.0 if rank <= 5.0 else 0.0,
    )


METRIC_FIELDS = ("reciprocal_rank", "wasted_effort", "exam", "hit1", "hit5")
