"""harness — runner, scorers, stats, report (Skill B, Phase 3). The heart.

    runner   thin model-in/out: HF checkpoint via transformers on GPU, OR GGUF
             via llama.cpp locally, OR a frontier API for the reference baseline.
             Temperature 0, fixed prompt templates, optional JSON-schema-
             constrained decoding (llama.cpp GBNF / outlines) as a togglable flag.
    scorers  hand-rolled ON PURPOSE (this is the learning): per-field exact match,
             micro/macro-F1, whole-record EM, JSON validity, hallucination rate
             (fabricated value where gold = null), abstention correctness.
    stats    bootstrap confidence intervals + McNemar paired test.
    report   -> reports/runs/<id>/report.md with the full reproducibility block
             (model hash, adapter hash, prompt-template id, decode params,
             eval-set hash) so any number can be re-derived.

Definition of done for Phase 3: `make harness MODEL=...` produces a scored,
reproducible report from a clean clone, twice, identically. The harness is
FROZEN before any training begins (§5 guardrail #2).
"""
