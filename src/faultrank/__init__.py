"""faultrank: offline spectrum-based fault localization and benchmarking.

Given a coverage spectrum (which statements each test executed, plus each
test's pass/fail outcome), rank program elements by how likely each is to
contain the fault, using spectrum-based fault localization (SBFL) formulas
(Tarantula, Ochiai, DStar, Jaccard, Op2, Barinel) and a coverage-aware
weighted heuristic. Everything runs offline on seeded synthetic data.
"""

from __future__ import annotations

from faultrank.formulas import FORMULAS, suspiciousness
from faultrank.heuristic import coverage_aware
from faultrank.ranking import fault_rank, rank_order
from faultrank.spectrum import Spectrum, SpectrumCounts

__all__ = [
    "FORMULAS",
    "Spectrum",
    "SpectrumCounts",
    "coverage_aware",
    "fault_rank",
    "rank_order",
    "suspiciousness",
]

__version__ = "0.1.0"
