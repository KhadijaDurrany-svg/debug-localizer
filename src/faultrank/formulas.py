"""Spectrum-based fault localization (SBFL) formulas.

Each formula maps the four per-element counts (ef, ep, nf, np) to a
suspiciousness score; higher means more suspicious. All functions are
vectorized over elements and never divide by zero -- degenerate elements
(for example, those executed by no test) score ``0.0``.

References
----------
* Jones & Harrold, "Empirical evaluation of the Tarantula automatic
  fault-localization technique" (ASE 2005).
* Abreu et al., "On the accuracy of spectrum-based fault localization"
  (TAICPART 2007) -- Ochiai, Jaccard.
* Wong et al., "The DStar method for effective software fault localization"
  (IEEE TR 2014).
* Naish et al., "A model for spectra-based software diagnosis"
  (TOSEM 2011) -- Op2.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from faultrank.spectrum import SpectrumCounts

_EPS = 1e-12


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Element-wise division that yields 0 where the denominator is ~0."""
    den = den.astype(np.float64)
    out = np.zeros_like(den, dtype=np.float64)
    nonzero = np.abs(den) > _EPS
    out[nonzero] = num.astype(np.float64)[nonzero] / den[nonzero]
    return out


def tarantula(c: SpectrumCounts) -> np.ndarray:
    """Tarantula (Jones & Harrold, 2005)."""
    fail_ratio = _safe_div(c.ef, np.full_like(c.ef, max(c.total_fail, 0)))
    pass_ratio = _safe_div(c.ep, np.full_like(c.ep, max(c.total_pass, 0)))
    return _safe_div(fail_ratio, fail_ratio + pass_ratio)


def ochiai(c: SpectrumCounts) -> np.ndarray:
    """Ochiai (Abreu et al., 2007)."""
    den = np.sqrt((c.total_fail * (c.ef + c.ep)).astype(np.float64))
    return _safe_div(c.ef.astype(np.float64), den)


def jaccard(c: SpectrumCounts) -> np.ndarray:
    """Jaccard coefficient."""
    return _safe_div(c.ef.astype(np.float64), (c.total_fail + c.ep).astype(np.float64))


def dstar(c: SpectrumCounts, star: int = 2) -> np.ndarray:
    """DStar / D* (Wong et al., 2014) with exponent ``star``.

    Defined as ``ef**star / (ep + nf)``. Where the denominator is zero but
    ``ef > 0`` the element is maximally suspicious; we assign it the largest
    finite score plus one so the ordering is well defined without infinities.
    """
    num = np.power(c.ef.astype(np.float64), float(star))
    den = (c.ep + c.nf).astype(np.float64)
    out = np.zeros_like(num)
    pos = den > _EPS
    out[pos] = num[pos] / den[pos]
    saturated = (~pos) & (c.ef > 0)
    if saturated.any():
        ceiling = float(out.max()) + 1.0 if out.size else 1.0
        out[saturated] = ceiling + num[saturated]
    return out


def op2(c: SpectrumCounts) -> np.ndarray:
    """Op2 (Naish et al., 2011): ``ef - ep / (total_pass + 1)``."""
    return c.ef.astype(np.float64) - c.ep.astype(np.float64) / (c.total_pass + 1.0)


def barinel(c: SpectrumCounts) -> np.ndarray:
    """Barinel ranking proxy: ``ef / (ef + ep)``."""
    return _safe_div(c.ef.astype(np.float64), (c.ef + c.ep).astype(np.float64))


FormulaFn = Callable[[SpectrumCounts], np.ndarray]

FORMULAS: dict[str, FormulaFn] = {
    "tarantula": tarantula,
    "ochiai": ochiai,
    "jaccard": jaccard,
    "dstar": dstar,
    "op2": op2,
    "barinel": barinel,
}


def suspiciousness(name: str, counts: SpectrumCounts) -> np.ndarray:
    """Look up and apply a formula by name."""
    try:
        fn = FORMULAS[name]
    except KeyError as exc:  # pragma: no cover - guarded by CLI/tests
        raise KeyError(f"unknown formula {name!r}; known: {sorted(FORMULAS)}") from exc
    return fn(counts)
