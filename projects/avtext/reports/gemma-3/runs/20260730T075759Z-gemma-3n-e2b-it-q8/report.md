# Run report — gemma-3n-E2B-it-Q8

- **Date:** 2026-07-30T07:57:59Z
- **Model:** `gemma-3n-E2B-it-Q8`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v1 · `sha256:608707987dfc8f13…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 620  ·  **JSON-valid:** 92.4% (47 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 620 | 51.4% [50.0, 52.8] | 10.4% | 0% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 200 | 52.3% [49.9, 54.7] | 6.3% | 0% |
| unseen_station | dissent | 60 | 48.4% [44.6, 52.1] | 7.5% | 0% |
| unseen_time | clean | 200 | 51.6% [48.9, 54.2] | 16.0% | 0% |
| unseen_time | dissent | 60 | 52.5% [48.0, 56.6] | 9.3% | 0% |
| unseen_time | parse_fail | 100 | 50.3% [45.7, 54.8] | 8.8% | 0% |

## Reproducibility
Every number above re-derives from:
- eval set `v1` — sha256 `608707987dfc8f139775021b48895fbfacba1c17c77e4bb733b1f0b61bd95aba`
- prompt `decode-json-v1`  ·  model `gemma-3n-E2B-it-Q8`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
