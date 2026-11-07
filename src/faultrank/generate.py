"""Seeded synthetic program & coverage-spectrum generator.

We model a program as a sequence of *blocks* (statements). Coverage and
failure are produced by a simple control-flow simulation, so the resulting
spectra have the structure that makes real fault localization hard -- not
random noise.

Model
-----
* A *mandatory spine* of blocks is executed by (almost) every test.
* Several *branch regions* are optional: a test enters each region with some
  probability. Within an entered region, each block is covered independently
  with probability ``q`` (so sibling statements are correlated but not
  identical -- this is what breaks exact ties and makes formulas differ).
* One block is the injected fault ``b*``, placed inside a region. A test
  **fails** iff it covers ``b*`` *and* the bug triggers (Bernoulli
  ``p_trigger``). ``p_trigger < 1`` models *coincidental correctness*: some
  tests execute the faulty line yet still pass, so the fault's passing-count
  is non-zero and precision is naturally capped below 1.
* Optional label noise flips a small fraction of outcomes, modelling flaky
  tests. Tests that cover no block are discarded.

The generator is fully determined by its seed and configuration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from faultrank.spectrum import Spectrum


@dataclass(frozen=True)
class GenConfig:
    """Configuration for one dataset of synthetic faulty programs."""

    n_programs: int = 200
    n_tests: int = 100
    spine_blocks: tuple[int, int] = (4, 8)  # inclusive range of mandatory blocks
    n_regions: tuple[int, int] = (3, 6)  # inclusive range of branch regions
    region_blocks: tuple[int, int] = (2, 5)  # inclusive blocks per region
    enter_prob: tuple[float, float] = (0.4, 0.85)  # per-region entry probability
    q: float = 0.95  # per-block coverage prob inside an entered region (tight siblings)
    p_trigger: tuple[float, float] = (0.25, 0.55)  # bug trigger prob (strong coincidental correctness)
    spine_skip: float = 0.03  # prob a spine block is (atypically) skipped
    label_noise: float = 0.0  # fraction of outcomes flipped (flaky-test robustness knob)
    min_failures: int = 3  # regenerate a program with fewer failing tests

    def __post_init__(self) -> None:
        for lo, hi in (self.spine_blocks, self.n_regions, self.region_blocks):
            if lo > hi or lo < 1:
                raise ValueError("range bounds must satisfy 1 <= lo <= hi")
        if not 0.0 < self.q <= 1.0:
            raise ValueError("q must be in (0, 1]")
        if not 0.0 <= self.label_noise < 0.5:
            raise ValueError("label_noise must be in [0, 0.5)")


@dataclass
class Program:
    """A generated faulty program and its test spectrum."""

    spectrum: Spectrum
    region_of: tuple[int, ...]  # region id per element (-1 for spine)
    config_digest: str = field(default="", repr=False)


def _rint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _generate_one(rng: np.random.Generator, cfg: GenConfig) -> Program:
    for _ in range(64):  # retry until the suite has enough failures
        n_spine = _rint(rng, *cfg.spine_blocks)
        n_regions = _rint(rng, *cfg.n_regions)
        region_sizes = [_rint(rng, *cfg.region_blocks) for _ in range(n_regions)]
        n_elems = n_spine + sum(region_sizes)

        # element layout: [spine blocks] then region blocks concatenated
        region_of = [-1] * n_spine
        region_slices: list[tuple[int, int]] = []
        cursor = n_spine
        for rsize in region_sizes:
            region_slices.append((cursor, cursor + rsize))
            region_of.extend([len(region_slices) - 1] * rsize)
            cursor += rsize

        enter_p = rng.uniform(cfg.enter_prob[0], cfg.enter_prob[1], size=n_regions)

        # pick the faulty region (prefer a region that is not entered too
        # often nor too rarely) and a block within it
        fault_region = int(rng.integers(0, n_regions))
        f_lo, f_hi = region_slices[fault_region]
        fault_idx = int(rng.integers(f_lo, f_hi))
        p_trigger = float(rng.uniform(*cfg.p_trigger))

        cov = np.zeros((cfg.n_tests, n_elems), dtype=bool)
        outcomes = np.zeros(cfg.n_tests, dtype=bool)

        for t in range(cfg.n_tests):
            # spine: covered unless an atypical skip
            spine_mask = rng.random(n_spine) >= cfg.spine_skip
            cov[t, :n_spine] = spine_mask
            # regions
            entered = rng.random(n_regions) < enter_p
            for r, (lo, hi) in enumerate(region_slices):
                if entered[r]:
                    inside = rng.random(hi - lo) < cfg.q
                    cov[t, lo:hi] = inside
            # failure logic
            if cov[t, fault_idx] and rng.random() < p_trigger:
                outcomes[t] = True

        # label noise
        if cfg.label_noise > 0:
            flip = rng.random(cfg.n_tests) < cfg.label_noise
            outcomes[flip] = ~outcomes[flip]

        # drop empty tests
        keep = cov.any(axis=1)
        cov = cov[keep]
        outcomes = outcomes[keep]

        if int(outcomes.sum()) >= cfg.min_failures and int((~outcomes).sum()) >= 1:
            spectrum = Spectrum(
                coverage=cov,
                outcomes=outcomes,
                faults=(fault_idx,),
                element_names=tuple(f"b{i}" for i in range(n_elems)),
            )
            return Program(spectrum=spectrum, region_of=tuple(region_of))
    raise RuntimeError("could not generate a program with enough failures; relax config")


def config_digest(cfg: GenConfig, seed: int) -> str:
    """Stable digest of the configuration + seed (not the data itself)."""
    h = hashlib.sha256()
    h.update(repr(cfg).encode())
    h.update(str(seed).encode())
    return h.hexdigest()[:16]


def generate_dataset(cfg: GenConfig, seed: int = 0) -> list[Program]:
    """Generate a deterministic list of faulty programs."""
    rng = np.random.default_rng(seed)
    digest = config_digest(cfg, seed)
    programs = []
    for _ in range(cfg.n_programs):
        prog = _generate_one(rng, cfg)
        prog.config_digest = digest
        programs.append(prog)
    return programs


def dataset_checksum(programs: list[Program]) -> str:
    """SHA-256 over every program's spectrum checksum, order-sensitive."""
    h = hashlib.sha256()
    for prog in programs:
        h.update(prog.spectrum.checksum().encode())
    return h.hexdigest()
