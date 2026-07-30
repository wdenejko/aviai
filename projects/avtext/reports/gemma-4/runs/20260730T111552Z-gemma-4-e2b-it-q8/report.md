# Run report — gemma-4-E2B-it-Q8

- **Date:** 2026-07-30T11:15:52Z
- **Model:** `gemma-4-E2B-it-Q8`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v1 · `sha256:608707987dfc8f13…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 620  ·  **JSON-valid:** 98.9% (7 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 620 | 69.5% [68.4, 70.6] | 31.0% | 2% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 200 | 67.5% [65.1, 69.8] | 26.7% | 2% |
| unseen_station | dissent | 60 | 64.4% [61.4, 67.1] | 32.1% | 0% |
| unseen_time | clean | 200 | 71.5% [69.7, 73.4] | 30.7% | 6% |
| unseen_time | dissent | 60 | 73.3% [70.9, 75.7] | 40.7% | 0% |
| unseen_time | parse_fail | 100 | 70.2% [68.7, 71.8] | 33.3% | 1% |

## Reproducibility
Every number above re-derives from:
- eval set `v1` — sha256 `608707987dfc8f139775021b48895fbfacba1c17c77e4bb733b1f0b61bd95aba`
- prompt `decode-json-v1`  ·  model `gemma-4-E2B-it-Q8`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
