from __future__ import annotations

import numpy as np

from faultrank.metrics import METRIC_FIELDS, score_program


def test_score_program_perfect():
    scores = np.array([0.9, 0.1, 0.1])
    ps = score_program(scores, (0,), 3)
    assert ps.rank == 1.0
    assert ps.reciprocal_rank == 1.0
    assert ps.wasted_effort == 0.0
    assert ps.hit1 == 1.0
    assert ps.hit5 == 1.0


def test_score_program_fields_present():
    scores = np.array([0.9, 0.1, 0.1])
    ps = score_program(scores, (1,), 3)
    for f in METRIC_FIELDS:
        assert hasattr(ps, f)


def test_hit5_boundary():
    scores = np.arange(10, dtype=float)[::-1]  # 9,8,...,0
    # fault at index 5 -> score 4.0 -> rank 6 -> not in top-5
    ps = score_program(scores, (5,), 10)
    assert ps.rank == 6.0
    assert ps.hit5 == 0.0
    # fault at index 4 -> score 5.0 -> rank 5 -> in top-5
    ps2 = score_program(scores, (4,), 10)
    assert ps2.hit5 == 1.0
