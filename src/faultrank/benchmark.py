"""Run every localizer over a synthetic dataset and report honest results.

The benchmark evaluates each method on every generated program, collects the
per-program metric vectors, and summarizes each metric with a mean and a
percentile bootstrap 95% CI. It then runs paired permutation tests comparing
the proposed coverage-aware heuristic against the baselines and against plain
Ochiai, on per-program reciprocal rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from faultrank import baselines
from faultrank.formulas import FORMULAS
from faultrank.generate import GenConfig, Program, dataset_checksum, generate_dataset
from faultrank.heuristic import coverage_aware
from faultrank.metrics import METRIC_FIELDS, score_program
from faultrank.stats import bootstrap_ci, paired_permutation_test

# Methods that beat the baselines are what we care about; the proposed
# contribution is "coverage_aware".
PROPOSED = "coverage_aware"
BASELINE_METHODS = ("random", "failcount", "last")


@dataclass
class MethodResult:
    name: str
    is_baseline: bool
    per_program: dict[str, np.ndarray] = field(default_factory=dict)
    summary: dict[str, dict[str, float]] = field(default_factory=dict)


def _method_scores(
    name: str, prog: Program, rng: np.random.Generator
) -> np.ndarray:
    counts = prog.spectrum.counts()
    if name in FORMULAS:
        return FORMULAS[name](counts)
    if name == PROPOSED:
        return coverage_aware(prog.spectrum)
    if name == "random":
        return baselines.random_scores(prog.spectrum, rng)
    if name == "failcount":
        return baselines.failcount(counts)
    if name == "last":
        return baselines.last_frame(prog.spectrum, counts)
    raise KeyError(f"unknown method {name!r}")


def run_benchmark(
    cfg: GenConfig | None = None,
    *,
    seed: int = 0,
    n_boot: int = 2000,
    n_perm: int = 10000,
) -> dict[str, Any]:
    """Execute the full benchmark and return a JSON-serializable result dict."""
    cfg = cfg or GenConfig()
    programs = generate_dataset(cfg, seed=seed)

    methods = [*FORMULAS.keys(), PROPOSED, *BASELINE_METHODS]
    results: dict[str, MethodResult] = {
        m: MethodResult(name=m, is_baseline=(m in BASELINE_METHODS)) for m in methods
    }

    # one RNG stream for the random baseline, seeded off the dataset seed
    rand_rng = np.random.default_rng(seed + 10_007)

    collected: dict[str, dict[str, list[float]]] = {
        m: {f: [] for f in METRIC_FIELDS} for m in methods
    }
    for prog in programs:
        n = prog.spectrum.n_elements
        for m in methods:
            scores = _method_scores(m, prog, rand_rng)
            ps = score_program(scores, prog.spectrum.faults, n)
            for f in METRIC_FIELDS:
                collected[m][f].append(getattr(ps, f))

    for m in methods:
        res = results[m]
        for f in METRIC_FIELDS:
            arr = np.asarray(collected[m][f], dtype=np.float64)
            res.per_program[f] = arr
            lo, hi = bootstrap_ci(arr, n_boot=n_boot, seed=seed + 1)
            res.summary[f] = {
                "mean": float(arr.mean()),
                "ci_lo": lo,
                "ci_hi": hi,
            }

    # significance: proposed vs baselines and vs ochiai, on reciprocal rank
    comparisons = []
    proposed_rr = results[PROPOSED].per_program["reciprocal_rank"]
    for other in (*BASELINE_METHODS, "ochiai"):
        other_rr = results[other].per_program["reciprocal_rank"]
        p = paired_permutation_test(proposed_rr, other_rr, n_perm=n_perm, seed=seed + 2)
        comparisons.append(
            {
                "proposed": PROPOSED,
                "versus": other,
                "metric": "reciprocal_rank",
                "mean_proposed": float(proposed_rr.mean()),
                "mean_versus": float(other_rr.mean()),
                "mean_diff": float((proposed_rr - other_rr).mean()),
                "p_value": p,
                "n_perm": n_perm,
            }
        )

    # rank methods by MRR for a quick leaderboard
    leaderboard = sorted(
        ((m, results[m].summary["reciprocal_rank"]["mean"]) for m in methods),
        key=lambda kv: kv[1],
        reverse=True,
    )

    return {
        "config": {
            "n_programs": cfg.n_programs,
            "n_tests": cfg.n_tests,
            "q": cfg.q,
            "label_noise": cfg.label_noise,
            "seed": seed,
            "n_boot": n_boot,
            "n_perm": n_perm,
        },
        "dataset_checksum": dataset_checksum(programs),
        "methods": {
            m: {"is_baseline": results[m].is_baseline, "metrics": results[m].summary}
            for m in methods
        },
        "leaderboard": [{"method": m, "mrr": v} for m, v in leaderboard],
        "comparisons": comparisons,
    }
