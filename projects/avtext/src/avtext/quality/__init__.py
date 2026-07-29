"""quality — tier-0 checks (Skill B, Phase 2).

The cheapest, most objective correctness signals, applied per record:

  - Round-trip property tests (Hypothesis): re-encode(decode(raw)) ≈ raw after
    canonicalization. If a decode can't reproduce its input, distrust it.
  - Invariant suite: dewpoint ≤ temp, gust > sustained wind, vocabulary ∈
    WMO/CCT tables, station ∈ registry, TAF period ordering sane, etc.

Failures are NEVER dropped — they route to the hard-case queue. Those messy
records are exactly the material the fine-tune must learn to handle (and to
*abstain* on when truly garbled).
"""
