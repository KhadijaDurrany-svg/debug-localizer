from __future__ import annotations

import numpy as np

from faultrank.generate import (
    GenConfig,
    config_digest,
    dataset_checksum,
    generate_dataset,
)


def test_deterministic_same_seed():
    cfg = GenConfig(n_programs=5, n_tests=40)
    a = generate_dataset(cfg, seed=7)
    b = generate_dataset(cfg, seed=7)
    assert dataset_checksum(a) == dataset_checksum(b)


def test_different_seed_differs():
    cfg = GenConfig(n_programs=5, n_tests=40)
    a = generate_dataset(cfg, seed=1)
    b = generate_dataset(cfg, seed=2)
    assert dataset_checksum(a) != dataset_checksum(b)


def test_fault_covered_by_every_failing_test():
    cfg = GenConfig(n_programs=10, n_tests=60)
    for prog in generate_dataset(cfg, seed=3):
        sp = prog.spectrum
        f = sp.faults[0]
        failing = sp.coverage[sp.outcomes]
        # by construction a test can only fail if it covered the fault
        assert np.all(failing[:, f])


def test_has_minimum_failures():
    cfg = GenConfig(n_programs=10, n_tests=60, min_failures=3)
    for prog in generate_dataset(cfg, seed=4):
        assert int(prog.spectrum.outcomes.sum()) >= 3
        assert int((~prog.spectrum.outcomes).sum()) >= 1


def test_coincidental_correctness_present():
    # at least some program should have a passing test covering the fault
    cfg = GenConfig(n_programs=20, n_tests=80)
    found = False
    for prog in generate_dataset(cfg, seed=5):
        sp = prog.spectrum
        f = sp.faults[0]
        passing = sp.coverage[~sp.outcomes]
        if passing[:, f].any():
            found = True
            break
    assert found, "expected coincidental correctness somewhere"


def test_config_digest_depends_on_seed():
    cfg = GenConfig()
    assert config_digest(cfg, 1) != config_digest(cfg, 2)


def test_region_labels_align_with_elements():
    cfg = GenConfig(n_programs=3, n_tests=30)
    for prog in generate_dataset(cfg, seed=6):
        assert len(prog.region_of) == prog.spectrum.n_elements


def test_no_empty_tests_remain():
    cfg = GenConfig(n_programs=5, n_tests=50)
    for prog in generate_dataset(cfg, seed=8):
        assert np.all(prog.spectrum.coverage.any(axis=1))


def test_label_noise_changes_dataset():
    clean = GenConfig(n_programs=6, n_tests=50, label_noise=0.0)
    noisy = GenConfig(n_programs=6, n_tests=50, label_noise=0.2)
    a = generate_dataset(clean, seed=9)
    b = generate_dataset(noisy, seed=9)
    assert dataset_checksum(a) != dataset_checksum(b)
