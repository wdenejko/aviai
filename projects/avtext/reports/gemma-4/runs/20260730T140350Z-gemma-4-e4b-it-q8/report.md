# Run report — gemma-4-E4B-it-Q8

- **Date:** 2026-07-30T14:03:50Z
- **Model:** `gemma-4-E4B-it-Q8`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v1 · `sha256:608707987dfc8f13…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 620  ·  **JSON-valid:** 100.0% (0 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 620 | 81.4% [80.5, 82.2] | 15.6% | 6% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 200 | 83.8% [82.1, 85.3] | 4.9% | 11% |
| unseen_station | dissent | 60 | 72.3% [69.5, 75.1] | 0.0% | 0% |
| unseen_time | clean | 200 | 83.9% [82.4, 85.3] | 13.9% | 6% |
| unseen_time | dissent | 60 | 82.2% [79.7, 84.7] | 1.9% | 3% |
| unseen_time | parse_fail | 100 | 76.4% [75.1, 77.7] | 42.1% | 0% |

## Reproducibility
Every number above re-derives from:
- eval set `v1` — sha256 `608707987dfc8f139775021b48895fbfacba1c17c77e4bb733b1f0b61bd95aba`
- prompt `decode-json-v1`  ·  model `gemma-4-E4B-it-Q8`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
