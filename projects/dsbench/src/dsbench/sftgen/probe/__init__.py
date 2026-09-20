"""The held-out generalisation probe (ADR-004 validation).

dsbench is the fine-tune's before/after metric AND the target the SFT data is built to move, so a
gain on dsbench alone cannot distinguish learning from teaching-to-the-test. This probe is a THIRD,
independent distribution: it exercises the SAME three skills (dialect date/time conventions,
denominator/population reasoning, agentic ML-delivery) on FRESH domains that appear neither in
dsbench (aviation) nor in the sftgen generators (retail/iot/support/payments/web/gym). The problems
are hand-written, not generator-produced, so they do not share the training data's shape.

The real generalisation claim is: the fine-tune moves BOTH the probe and dsbench. A dsbench gain
without a probe gain is an overfitting alarm (ADR-004 §Validation, kill criterion). The problems are
oracle-gated by `selftest()` (setup -> reference -> check, no model) exactly like the eval set, and
run through the same agentic harness with a NEUTRAL system prompt (see `runner.py`).
"""
