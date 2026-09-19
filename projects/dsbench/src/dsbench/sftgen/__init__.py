"""sftgen -- targeted SFT-data generators for the Qwen3.6-35B-A3B fine-tune (ADR-004).

The sandbox that MEASURES the model (ADR-003) also MANUFACTURES its training data here, under the
same discipline: every row carries an execution proof. Three generators, one per Gate-0-measured
gap (ADR-004):

  - Target A  `dialect_conventions`  -- SQL-dialect date/time conventions (weekday numbering,
                                        timezone direction). Own-generated, teacher-free, verified
                                        against real engines across dialects. Apache-2.0-clean.
  - Target B  `denominator_reasoning` (todo) -- conditional-population / ratio-denominator sense.
  - Target C  `ml_delivery_trajectories` (todo) -- agentic ML-workflow delivery discipline.

Design invariants shared by all three (ADR-004):
  * teach the TRANSFERABLE skill on a distribution that is NOT the aviation benchmark, so the gain
    generalises rather than memorising dsbench;
  * keep only EXECUTION-VERIFIED rows (a pandas truth and an engine result must agree), so quality
    is proven, not hoped, and provenance stays clean;
  * every row records its `provenance` and `verification` so the ADR-001 Gate-3 licence audit and
    the decontamination gate are mechanical.
"""
