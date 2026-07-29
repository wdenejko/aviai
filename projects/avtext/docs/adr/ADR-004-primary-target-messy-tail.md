# ADR-004: The primary target is the messy tail + abstention, not clean decode

- **Status:** Accepted
- **Date:** 2026-07-29
- **Supersedes emphasis in:** ADR-001 (which scoped *which data*; this scopes
  *what "better" means*)

## Context

Both companion briefings in `docs/research/` reach the same load-bearing
conclusion, from different angles:

- **Clean METAR/TAF decoding is a solved, deterministic problem.** Mature
  parsers (`python-metar`, `avwx-engine`, `metar-taf-parser`, `metaf`) hit
  ~99–100% field accuracy on well-formed reports. Fine-tuning a 4.5B model to do
  this is "re-solving a solved problem with a worse tool" — slower, costlier,
  less reliable than 200 lines of parser.
- **The LLM's defensible headroom is where parsers fail:** malformed/garbled
  AUTO observations, typos, truncation, non-standard national idioms, the
  free-text US `RMK` section, fluent plain-language briefing, anomaly detection,
  and — critically for a safety domain — **knowing when to abstain** instead of
  fabricating a value.
- Evidence this is fine-tuning's sweet spot: *LoRA Land* (small fine-tuned models
  beat GPT-4 on structured extraction/NER, lose on open-ended generation); the
  *Pre-Flight* benchmark (~8B ≈ 58% vs frontier ~80–83% on multi-step reasoning —
  so lean on format/extraction, not reasoning). And *Gekhman et al. 2024*:
  training on facts not derivable from the input manufactures hallucination.

If we made **clean decode** the headline, a near-ceiling base model would leave
almost no room to move, and fine-tuning would look like it "did nothing" — a
ceiling effect that hides the real signal.

## Decision

The project's **headline metrics** are, in order:

1. **Messy-tail field macro-F1** (macro, so rare-token regressions surface).
2. **Hallucination rate** — fabricated value where gold = null.
3. **Abstention correctness** — garbled input flagged, not guessed.
4. Faithful plain-language briefing (facts diffed against the authoritative
   decode) — a secondary generative target.

**Clean decode is kept only as a reference row** (the deterministic parser is the
ceiling there). Training data must include explicit **abstention** examples and
**corruption-augmented** hard cases, and must **never** include prediction or
unstated-fact targets (ADR-001 guardrail #5, now backed by Gekhman et al.).

Deployment shape is **hybrid**: parser owns clean input + acts as label
oracle/verifier; fine-tuned E4B owns the messy tail + natural language.

## Consequences

- **+** The before/after headline measures something with real headroom, so a
  genuine improvement won't be masked by a ceiling.
- **+** Directly serves the safety posture (regulators treat AI weather output as
  advisory): we measure *not-lying* and *knowing-when-to-quit*, not just accuracy.
- **−** Requires the hard-case queue and corruption augmentation to be first-class
  from Phase 1, and needs enough messy items per stratum for tight CIs
  (~several hundred–1–2k per stratum, per the finetune briefing §5.5).
- Shapes station selection (favor AUTO-heavy / terrain / convective — already in
  `configs/stations.yaml`) and the eval-set builders (`tasks/`).

## Alternatives considered

- *Clean decode as the headline (the literal original framing).* Rejected: the
  parser already wins there and the base model is near-ceiling, so the metric
  can't reveal fine-tuning's effect. Retained as a reference only.
- *Chase cross-report / route reasoning as primary.* Rejected for now: highest
  value but needs frontier-class reasoning; a 4.5B model is weakest here
  (Pre-Flight). Revisit post-slice.
