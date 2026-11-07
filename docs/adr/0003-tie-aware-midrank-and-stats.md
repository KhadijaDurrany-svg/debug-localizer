# 3. Mid-rank tie policy and resampling statistics

Status: accepted

## Context

SBFL scores produce many ties, and how ties are resolved can swing reported
accuracy dramatically (best-case vs worst-case ranking). Point estimates
alone also hide sampling uncertainty.

## Decision

* Use the **mid-rank** (average-rank) tie policy everywhere: a fault tied
  with `k-1` other elements at the top gets rank `(k+1)/2`. This is the
  expected inspection position under random tie-breaking -- neither
  optimistic nor pessimistic.
* Report every metric with a **percentile bootstrap 95% CI** over programs.
* Compare the proposed heuristic against baselines with a **paired
  permutation (sign-flip) test** on per-program reciprocal rank, with
  add-one smoothing so a p-value is never exactly zero.

## Consequences

* Metrics are directly comparable across methods and honest about ties.
* Results carry error bars and significance, not bare means.
* Permutation and bootstrap are seeded, so the whole report is reproducible.
