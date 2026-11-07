# 1. Spectrum-based localization, not LLM code review

Status: accepted

## Context

The problem space is "automated debugging". A common approach is to feed
source files to a large language model and ask it to spot bugs. That path
needs network access, API keys and a non-reproducible external service, and
it localizes by *reading* code rather than by *evidence from executions*.

## Decision

faultrank localizes faults from **execution spectra**: the record of which
program elements each test executed, together with pass/fail outcomes. It
implements classic spectrum-based fault localization (SBFL) formulas plus a
coverage-aware heuristic. No model, no network, no keys.

## Consequences

* Fully offline and reproducible; results depend only on inputs and a seed.
* The engine is language-agnostic -- it consumes a coverage matrix, not
  source text, so any tracer that emits one can feed it.
* It cannot reason about code semantics the way an LLM reviewer can; it only
  ranks elements by statistical association with failure. We treat that as a
  feature (evidence-based, auditable) and document the ceiling honestly.
