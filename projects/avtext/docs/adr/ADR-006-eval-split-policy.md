# ADR-006: eval/v1 split policy — station + time holdout, with a data-forced asymmetry

- **Status:** Accepted
- **Date:** 2026-07-29
- **Builds on:** ADR-004 (the tail is the target), ADR-003 (Gemma-4-E4B, Jan-2025 cutoff)

## Context

Phase 3 freezes `eval/v1` before any training (the harness guardrail: never tune against
the eval). The set must be held out from whatever we later train on, and the *axis* of
holdout defines what "generalization" the study measures. Three options were on the table:
time-only (new dates, seen stations), station-only (new stations/formats), or both. We
chose **both** — the strictest — so the eval reports two distinct generalizations
separately: `unseen_station` and `unseen_time`.

Building it surfaced three empirical facts about the corpus that reshaped the design. None
were visible from the station config; all came from measuring the mined pool.

1. **US-volume dominance.** A global-random sample of recent data is swamped by
   high-frequency US airports (all clean). The scan must be **station-stratified** or the
   rare tail stations never appear.
2. **The tail is concentrated, and the config tags were wrong.** `stations.yaml` tagged
   Scandinavia as the parse-fail source; in fact parse failures come overwhelmingly from
   **UUWW** (Moscow Vnukovo, ~38%) and, historically, **YSSY** (Sydney). Dissent comes from
   the US RMK T-group fields (KAIG/KOZW/KEKM).
3. **Parse failures are non-stationary in time.** YSSY failed at ~53% in 2023, then its
   format was fixed at **2025-02** and it now parses clean (0%). UUWW went the other way:
   37% → **74%** across the 2026-05 boundary. Post-cutoff, UUWW is the *only* meaningful
   parse-fail source; every other station is <1%.

Fact 3 is the binding one. With a single post-cutoff abstention source, we cannot put
parse-fail cases in *both* a held-out station and the recent-time pool — there is only one
well to draw from.

## Decision

- **Both axes, disjoint by construction** (`harness/freeze.py::assign_split`):
  - `unseen_station` = a held-out station (any time ≥ cutoff floor).
  - `unseen_time` = a training station, on/after `2026-05-01`.
  - `train` = training station, before the boundary (reserved for Phase 4).
- **Held-out stations `{KEKM, EGCC, KSEA, EPKK}`**, chosen from the *measured* distribution:
  KEKM supplies the dissent tail (T-group; siblings KAIG/KOZW stay in training so the format
  is learnable), the rest are clean EU/US/PL baselines with same-family siblings retained.
- **UUWW stays in training** and its abstention test is carried by **time-holdout**: we
  train on its pre-boundary failures and score its post-boundary failures in `unseen_time`.
  This is the cleanest possible abstention experiment — same station, temporally disjoint,
  real (not synthetic) failures, and a genuine distribution shift (37% → 74%).
- **Asymmetric per-stratum targets** (a ceiling, never padded): `unseen_station` =
  {200 clean, 60 dissent}; `unseen_time` = {200 clean, 100 parse_fail, 60 dissent}. The
  asymmetry is fact 3 made concrete, not a modelling choice.
- **Memorization floor `2025-02-01`** on every item (after ADR-003's Jan-2025 cutoff).
- **Frozen** = deterministic (md5-ordered, station-stratified sampling) + content-hashed;
  same corpus + config ⇒ identical `sha256`. Verified by freezing twice.

## Consequences

- **+** Two clean generalization axes; abstention has a real, temporally-disjoint train/test
  split (UUWW), which is exactly what Phase 4 needs to measure abstention improvement.
- **+** The freeze is reproducible and self-describing (manifest carries every parameter).
- **−** No abstention signal in `unseen_station`, and the `unseen_time` parse-fail bucket is
  UUWW-dominated — a single-station, single-format abstention test. Honest, but narrow.
- **−** Holding out whole stations costs their volume from training (acceptable: 47 remain).
- **Risk / v2:** the tail rests on very few stations. If UUWW's format normalizes like
  YSSY's did, the abstention source could vanish — widen the station set, or supplement with
  mutant-injected (synthetic) abstention cases, before relying on it for a headline result.
