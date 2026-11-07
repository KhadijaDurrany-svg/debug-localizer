from __future__ import annotations

import numpy as np
import pytest

from faultrank.ranking import exam_score, fault_rank, rank_order, wasted_effort


def test_fault_rank_top():
    scores = np.array([0.1, 0.9, 0.3])
    assert fault_rank(scores, (1,)) == 1.0


def test_fault_rank_midrank_on_ties():
    # three-way tie at the top -> mid-rank (3+1)/2 = 2.0
    scores = np.array([0.5, 0.5, 0.5])
    assert fault_rank(scores, (0,)) == 2.0


def test_fault_rank_two_way_tie_at_top():
    scores = np.array([0.9, 0.9, 0.1])
    # two tied at top: mid-rank = 0 strictly-greater + (2+1)/2 = 1.5
    assert fault_rank(scores, (0,)) == 1.5


def test_fault_rank_last():
    scores = np.array([0.9, 0.8, 0.1])
    assert fault_rank(scores, (2,)) == 3.0


def test_wasted_effort_matches_rank():
    scores = np.array([0.9, 0.8, 0.1])
    assert wasted_effort(scores, (2,)) == 2.0


def test_exam_score_fraction():
    scores = np.array([0.9, 0.8, 0.1, 0.0])
    assert exam_score(scores, (0,), 4) == pytest.approx(0.25)


def test_multiple_faults_uses_best():
    scores = np.array([0.1, 0.9, 0.3])
    # best of faults {0,1} is element 1 at rank 1
    assert fault_rank(scores, (0, 1)) == 1.0


def test_rank_order_descending_with_index_tiebreak():
    scores = np.array([0.5, 0.9, 0.5])
    order = rank_order(scores)
    assert order[0] == 1
    # ties (0 and 2) broken by ascending index
    assert list(order[1:]) == [0, 2]


def test_fault_rank_requires_indices():
    with pytest.raises(ValueError):
        fault_rank(np.array([0.1, 0.2]), ())
