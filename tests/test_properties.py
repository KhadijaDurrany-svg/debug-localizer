"""Property-based tests (Hypothesis) over random valid spectra."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from faultrank.formulas import FORMULAS, barinel, jaccard, ochiai, tarantula
from faultrank.heuristic import coverage_aware
from faultrank.ranking import fault_rank
from faultrank.spectrum import Spectrum

BOUNDED = (ochiai, tarantula, jaccard, barinel)


@st.composite
def spectra(draw, max_tests=12, max_elems=8):
    n_tests = draw(st.integers(min_value=2, max_value=max_tests))
    n_elems = draw(st.integers(min_value=1, max_value=max_elems))
    rows = []
    for _ in range(n_tests):
        row = draw(
            st.lists(
                st.booleans(), min_size=n_elems, max_size=n_elems
            )
        )
        # guarantee the test covers something so no empty-row surprises
        if not any(row):
            row[draw(st.integers(0, n_elems - 1))] = True
        rows.append(row)
    cov = np.array(rows, dtype=bool)
    # at least one failing and one passing test
    outcomes = draw(
        st.lists(st.booleans(), min_size=n_tests, max_size=n_tests)
    )
    if not any(outcomes):
        outcomes[0] = True
    if all(outcomes):
        outcomes[-1] = False
    out = np.array(outcomes, dtype=bool)
    fault = draw(st.integers(min_value=0, max_value=n_elems - 1))
    return Spectrum(coverage=cov, outcomes=out, faults=(fault,))


@settings(max_examples=200, deadline=None)
@given(spectra())
def test_counts_invariants(sp):
    c = sp.counts()
    assert np.all(c.ef + c.nf == c.total_fail)
    assert np.all(c.ep + c.np_ == c.total_pass)
    assert np.all(c.ef >= 0) and np.all(c.ep >= 0)


@settings(max_examples=200, deadline=None)
@given(spectra())
def test_formulas_finite_and_non_nan(sp):
    counts = sp.counts()
    for name, fn in FORMULAS.items():
        scores = fn(counts)
        assert scores.shape == (sp.n_elements,)
        assert np.all(np.isfinite(scores)), name
        assert not np.any(np.isnan(scores)), name


@settings(max_examples=200, deadline=None)
@given(spectra())
def test_bounded_formulas_in_unit_interval(sp):
    counts = sp.counts()
    for fn in BOUNDED:
        scores = fn(counts)
        assert np.all(scores >= -1e-9)
        assert np.all(scores <= 1.0 + 1e-9)


@settings(max_examples=200, deadline=None)
@given(spectra())
def test_coverage_aware_finite_non_negative(sp):
    scores = coverage_aware(sp)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= -1e-12)


@settings(max_examples=200, deadline=None)
@given(spectra())
def test_fault_rank_within_bounds(sp):
    counts = sp.counts()
    scores = ochiai(counts)
    rank = fault_rank(scores, sp.faults)
    assert 1.0 <= rank <= sp.n_elements


@settings(max_examples=150, deadline=None)
@given(spectra())
def test_element_never_executed_scores_zero(sp):
    # append a column executed by no test; it must score 0 for ratio formulas
    cov = np.hstack([sp.coverage, np.zeros((sp.n_tests, 1), dtype=bool)])
    sp2 = Spectrum(coverage=cov, outcomes=sp.outcomes, faults=(0,))
    counts = sp2.counts()
    for fn in (ochiai, tarantula, jaccard, barinel):
        assert fn(counts)[-1] == 0.0
