# 2. A control-flow-aware synthetic generator

Status: accepted

## Context

Honest benchmarking needs ground truth (where the bug really is) and a
difficulty profile that resembles real fault localization. Sampling coverage
matrices from independent coin flips would make the task trivially easy: the
faulty element would stand alone and every formula would score ~100%.

## Decision

Generate programs with a mandatory spine plus optional *branch regions*. The
fault lives inside a region alongside sibling statements that are covered
*together but not identically* (per-block dropout `q`). Failure requires
covering the fault **and** a Bernoulli trigger (`p_trigger < 1`), modelling
*coincidental correctness*. This yields:

* confounders (siblings) that tie with or nearly tie with the true fault, so
  top-1 accuracy is naturally below 1.0;
* passing tests that execute the fault, so precision is capped;
* optional label noise for flaky-test robustness checks.

## Consequences

* Accuracy numbers sit in a realistic mid-range, not a suspicious ~1.0.
* The generator exposes knobs (`q`, `p_trigger`, `label_noise`) used to
  *harden* the benchmark if results look too easy.
* It is synthetic: absolute numbers should not be read as real-world
  effectiveness, only as relative comparisons between methods. This is called
  out under "Known gaps" in the README.
