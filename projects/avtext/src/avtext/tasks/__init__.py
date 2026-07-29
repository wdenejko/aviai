"""tasks — eval-set builders (Skill B, Phase 3).

Turn the consensus-labeled corpus into *frozen* evaluation sets under
eval/v1/*.jsonl, each with a content hash.

Planned slices:
    decode->JSON (clean)     well-formed METARs
    decode->JSON (hard-case) messy / AUTO / malformed tails
    flight-category          derived VFR/MVFR/IFR/LIFR classification
    external held-out        Pre-Flight, ERAU (eval-only)

Split discipline is ENFORCED IN CODE, not by convention:
  - test stations never appear in train (station split)
  - test observations are strictly AFTER the model's Jan-2025 cutoff (time split)
This module owns that logic so finetune/prep.py can reuse it — leakage
prevention as code, in one place.
"""
