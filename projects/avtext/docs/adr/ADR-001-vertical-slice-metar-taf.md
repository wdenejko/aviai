# ADR-001: Vertical slice = METAR/TAF, one task family, before anything widens

- **Status:** Accepted
- **Date:** 2026-07-29

## Context

Aviation text spans ~8 use cases (METAR, TAF, NOTAM, PIREP, SIGMET/AIRMET, …).
It is tempting to build broad ingestion and a general schema up front. But this
is a *learning* project whose hardest, newest skills are the **eval harness** and
**fine-tuning** — not breadth of data. A complete dataset feeding a harness that
doesn't exist yet teaches nothing; a working end-to-end loop teaches everything.

## Decision

Take a **thin vertical slice** — METAR/TAF only, ~50 stations, a single task
family (raw → structured JSON, plus derived flight-category) — all the way
through: download → schema → oracles → harness → baseline → fine-tune →
before/after report. **No NOTAM/PIREP/SIGMET code is written until that slice
produces a committed report.**

## Consequences

- **+** Fastest path to the actual learning (harness design, fine-tuning).
- **+** Every later use case reuses proven schema/oracle/harness patterns.
- **−** Some ingestion code will be re-generalized later. Accepted — cheap
  relative to the cost of building breadth against an unproven loop.
- **Guardrail:** the 8-use-case map is the roadmap, not the sprint (plan §5.1).

## Alternatives considered

- *Breadth-first ingestion of all products.* Rejected: defers the real learning
  and risks a large dataset for a loop that may need reshaping.
- *METAR only (drop TAF).* Rejected: TAF adds forecast-period structure that
  meaningfully exercises the schema without widening sources.
