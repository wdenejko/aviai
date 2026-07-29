"""consensus — tier-1 labeling (Skill B, Phase 2).

Turn several fallible oracles into one label per field.

  v1: field-level majority vote across the 3 parsers + AWC decoded JSON as a 4th
      voice. Any disagreement -> hard-case queue.
  v2 (later): Dawid–Skene latent-truth estimation, compared against v1 — a
      self-contained learning experiment in "who do you trust when they differ".

KPI to track: panel agreement (Krippendorff's α). Consensus is only trustworthy
*after* calibration against the gold seed + a mutant-injection audit (plant known
errors; confirm the panel catches them). "Three parsers agree" can be three
parsers sharing a bug.
"""
