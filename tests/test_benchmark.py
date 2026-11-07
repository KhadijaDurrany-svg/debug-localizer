from __future__ import annotations

from faultrank.benchmark import BASELINE_METHODS, PROPOSED, run_benchmark
from faultrank.generate import GenConfig


def _small():
    return run_benchmark(
        GenConfig(n_programs=40, n_tests=60), seed=0, n_boot=200, n_perm=500
    )


def test_benchmark_runs_and_has_all_methods():
    res = _small()
    assert PROPOSED in res["methods"]
    for b in BASELINE_METHODS:
        assert b in res["methods"]
    assert "ochiai" in res["methods"]


def test_benchmark_deterministic():
    a = _small()
    b = _small()
    assert a["dataset_checksum"] == b["dataset_checksum"]
    assert a["leaderboard"] == b["leaderboard"]


def test_sbfl_beats_random():
    res = _small()
    mrr = {m: res["methods"][m]["metrics"]["reciprocal_rank"]["mean"] for m in res["methods"]}
    assert mrr["ochiai"] > mrr["random"]
    assert mrr[PROPOSED] > mrr["random"]


def test_comparisons_have_pvalues():
    res = _small()
    assert res["comparisons"]
    for comp in res["comparisons"]:
        assert 0.0 < comp["p_value"] <= 1.0
        assert comp["proposed"] == PROPOSED


def test_metrics_have_cis():
    res = _small()
    m = res["methods"]["ochiai"]["metrics"]["reciprocal_rank"]
    assert m["ci_lo"] <= m["mean"] <= m["ci_hi"]


def test_leaderboard_sorted_descending():
    res = _small()
    vals = [row["mrr"] for row in res["leaderboard"]]
    assert vals == sorted(vals, reverse=True)
