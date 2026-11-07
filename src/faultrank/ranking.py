"""Turn suspiciousness scores into ranks and localization effort.

Ties matter a lot in SBFL: sibling statements on the same branch often get
identical scores. We use the *mid-rank* (a.k.a. average-rank) tie policy,
which is the expected inspection position under uniform random tie-breaking:

    rank(e) = (# elements scoring strictly higher) + (# tied with e + 1) / 2

counting ``e`` itself among the ties. This is standard in the SBFL
literature and avoids the optimism of best-case ("fault is first among
ties") or the pessimism of worst-case ("fault is last among ties").
"""

from __future__ import annotations

import numpy as np


def fault_rank(scores: np.ndarray, fault_indices: tuple[int, ...] | list[int]) -> float:
    """Mid-rank (1-based) of the best-scoring ground-truth faulty element."""
    if not len(fault_indices):
        raise ValueError("need at least one fault index")
    scores = np.asarray(scores, dtype=np.float64)
    best = max(float(scores[i]) for i in fault_indices)
    strictly_greater = int((scores > best + 1e-12).sum())
    tied = int((np.abs(scores - best) <= 1e-12).sum())
    return strictly_greater + (tied + 1) / 2.0


def rank_order(scores: np.ndarray) -> np.ndarray:
    """Return element indices sorted by descending suspiciousness.

    Ties are broken by ascending element index, giving a deterministic
    inspection order for display purposes.
    """
    scores = np.asarray(scores, dtype=np.float64)
    return np.lexsort((np.arange(scores.size), -scores))


def wasted_effort(scores: np.ndarray, fault_indices: tuple[int, ...] | list[int]) -> float:
    """Expected number of non-faulty elements inspected before the fault."""
    return fault_rank(scores, fault_indices) - 1.0


def exam_score(
    scores: np.ndarray, fault_indices: tuple[int, ...] | list[int], n_elements: int
) -> float:
    """EXAM: fraction of elements inspected to reach the fault."""
    if n_elements <= 0:
        raise ValueError("n_elements must be positive")
    return fault_rank(scores, fault_indices) / n_elements
