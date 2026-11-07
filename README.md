# faultrank

**Offline spectrum-based fault localization (SBFL) with honest benchmarks.**

Given a *coverage spectrum* — which program elements each test executed, plus
each test's pass/fail outcome — `faultrank` ranks the elements by how likely
each is to contain the bug. It implements the classic SBFL formulas
(Tarantula, Ochiai, DStar, Jaccard, Op2, Barinel) and a **coverage-aware
weighted heuristic**, and it benchmarks localization accuracy against random
and naive baselines on seeded synthetic programs with injected faults.

Everything runs **offline**: no network, no API keys, no model downloads. The
only runtime dependency is NumPy. Results are fully reproducible from a seed.

## How this differs from an LLM code-review agent

A common "automated debugging" approach feeds source files to a language
model and asks it to find bugs — it needs network access and keys, is
non-reproducible, and localizes by *reading* code. `faultrank` localizes from
*execution evidence* instead: it consumes a coverage matrix and outcome
vector and ranks elements by statistical association with failure. It is
model-free, language-agnostic (anything that emits coverage can feed it),
deterministic, and auditable. It also differs from a CI build-log triage tool:
there is no log parsing here at all — the unit of analysis is a statement's
coverage across a test suite, not a line of build output. See
`docs/adr/0001-spectrum-based-over-llm-review.md`.

## Results

Measured by running `faultrank bench --seed 0` (200 synthetic faulty
programs, 100 tests each; default generator settings). All numbers below are
real output from this repository, not hand-written. Reproduce with the same
command; the run also writes `benchmarks/results.json` and a `manifest.json`
recording the git SHA, Python/NumPy versions and dataset checksum.

Metrics: **MRR** mean reciprocal rank of the fault; **hit@1 / hit@5** fraction
of programs where the fault's (mid-)rank is ≤ 1 / ≤ 5; **wasted** mean number
of non-faulty elements inspected before reaching the fault. Ties use the
mid-rank policy (`docs/adr/0003-...`). Higher MRR/hit is better; lower wasted
is better. `coverage_aware` is the proposed method; `*` marks baselines.

| method           | MRR   | 95% CI (MRR)     | hit@1 | hit@5 | wasted |
|------------------|-------|------------------|-------|-------|--------|
| **coverage_aware** | **0.878** | [0.846, 0.907] | **0.765** | 1.000 | **0.27** |
| ochiai           | 0.864 | [0.831, 0.892]   | 0.705 | 1.000 | 0.28   |
| dstar            | 0.864 | [0.831, 0.892]   | 0.705 | 1.000 | 0.28   |
| op2              | 0.864 | [0.831, 0.892]   | 0.705 | 1.000 | 0.28   |
| jaccard          | 0.851 | [0.818, 0.881]   | 0.685 | 1.000 | 0.31   |
| tarantula        | 0.816 | [0.780, 0.849]   | 0.635 | 1.000 | 0.44   |
| barinel          | 0.816 | [0.780, 0.849]   | 0.635 | 1.000 | 0.44   |
| `*` failcount    | 0.389 | [0.368, 0.410]   | 0.010 | 0.975 | 1.94   |
| `*` last         | 0.230 | [0.195, 0.267]   | 0.085 | 0.330 | 7.67   |
| `*` random       | 0.177 | [0.150, 0.208]   | 0.045 | 0.245 | 10.40  |

**Significance** (paired permutation test, sign-flip, 10,000 permutations, on
per-program reciprocal rank; add-one smoothing so p is never exactly 0):

| coverage_aware vs | mean diff | p-value |
|-------------------|-----------|---------|
| random            | +0.7005   | 0.0001  |
| failcount         | +0.4891   | 0.0001  |
| last              | +0.6474   | 0.0001  |
| ochiai            | +0.0143   | 0.0190  |

The heuristic beats every baseline by a wide, highly significant margin, and
it edges plain Ochiai by a small but statistically significant amount
(p ≈ 0.02) in this clean setting.

### Robustness: flaky tests

Re-running with 5% label noise (`faultrank bench --seed 0 --label-noise 0.05`)
degrades everyone, but the ordering is instructive (real output):

| method           | MRR   | hit@1 | wasted |
|------------------|-------|-------|--------|
| **coverage_aware** | **0.837** | 0.720 | 0.50 |
| ochiai           | 0.828 | 0.675 | 0.51   |
| jaccard          | 0.818 | 0.650 | 0.45   |
| tarantula        | 0.791 | 0.620 | 0.57   |
| op2              | 0.305 | 0.130 | 4.66   |
| `*` failcount    | 0.202 | 0.010 | 5.46   |

Count-based methods (Op2, failcount) collapse under label noise because they
lean on raw `ef`; the normalized formulas and the coverage-aware heuristic
stay robust. Under noise the heuristic's edge over Ochiai is no longer
significant (p ≈ 0.17) — reported honestly, not hidden.

## The coverage-aware heuristic

Plain SBFL treats every test equally. But a *failing* test that executes only
a few statements implicates each of them far more strongly than one that
executes half the program. The heuristic weights each test by

    w_t = 1 / (1 + log(k_t))        # k_t = number of elements test t covers

and computes an Ochiai-style score on the weighted counts. It reduces exactly
to Ochiai when all tests cover the same number of elements (there is a test
asserting this). Full rationale in `src/faultrank/heuristic.py`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # dev extras: pytest, hypothesis, ruff
```

Requires Python ≥ 3.10.

## Use

```bash
# run the benchmark (writes benchmarks/results.json + manifest.json)
faultrank bench --seed 0

# run without writing anything
faultrank bench --seed 0 --no-write

# emit one synthetic spectrum, then localize it with a chosen formula
faultrank gen --seed 1 --out /tmp/spec.json
faultrank rank /tmp/spec.json --formula coverage_aware --top 10
```

As a library:

```python
import numpy as np
from faultrank import Spectrum, coverage_aware, fault_rank

cov = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=bool)
outcomes = np.array([1, 1, 0, 0], dtype=bool)   # True == failing test
sp = Spectrum(coverage=cov, outcomes=outcomes, faults=(1,))
scores = coverage_aware(sp)
print(scores, fault_rank(scores, sp.faults))
```

## How the synthetic data is built

Programs are generated by a small control-flow simulation, not by independent
coin flips (which would be trivially easy). A mandatory spine runs in almost
every test; optional *branch regions* are entered probabilistically; the fault
sits inside a region among sibling statements covered *together but not
identically*, so confounders tie or nearly tie with the true fault. A test
fails only if it covers the fault **and** a Bernoulli trigger fires
(`p_trigger < 1`), modelling *coincidental correctness* — some tests execute
the buggy line yet still pass. Knobs (`q`, `p_trigger`, `label_noise`) let you
harden the benchmark; the defaults were chosen so accuracy is realistic (hit@1
≈ 0.6–0.8), not a suspicious ~1.0. See `docs/adr/0002-...`.

## Layout

```
src/faultrank/
  spectrum.py    # Spectrum data model + ef/ep/nf/np counts + checksum
  formulas.py    # Tarantula, Ochiai, DStar, Jaccard, Op2, Barinel
  heuristic.py   # coverage-aware weighted-Ochiai ranker (the proposed method)
  ranking.py     # mid-rank tie policy, wasted effort, EXAM
  metrics.py     # per-program MRR / hit@k / wasted effort
  generate.py    # seeded synthetic program + spectrum generator
  baselines.py   # random, failcount, last-frame baselines
  stats.py       # bootstrap CI + paired permutation test
  benchmark.py   # orchestration
  cli.py         # argparse CLI: bench / gen / rank
tests/           # 71 tests (pytest + Hypothesis)
docs/adr/        # architecture decision records
benchmarks/      # results.json + manifest.json (written by `bench`)
```

## Known gaps

- **Synthetic, not real programs.** Absolute numbers reflect the generator,
  not any real codebase. The benchmark is for *relative* comparison of methods
  under controlled, documented difficulty — not a claim about real-world bug
  localization rates. Validating on a real defect dataset (e.g. a Defects4J-
  style corpus) is future work and is intentionally out of scope for an
  offline, dependency-light repo.
- **hit@5 saturates at ~1.0** for all SBFL methods because the generated
  programs are small (≈15–35 elements) and the fault's basic block reliably
  lands in the top 5. The discriminating signal is in **hit@1** and **wasted
  effort**, which is where the methods actually separate.
- **The heuristic's edge over Ochiai is small** (ΔMRR ≈ 0.014) and only
  significant in the clean setting; under label noise it is within noise of
  Ochiai. It is a modest, honestly-measured refinement, not a breakthrough.
- **Single-fault model.** Each program has exactly one injected fault; multi-
  fault localization (and fault interaction) is not modelled.
- **Statements only.** The engine ranks whatever elements the coverage matrix
  defines (statements/lines/functions); it does not itself instrument code.

## License

MIT © 2026 Khadija. See `LICENSE`.
