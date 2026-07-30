# Run report — gemma-3-1b-it-Q8

- **Date:** 2026-07-30T07:27:48Z
- **Model:** `gemma-3-1b-it-Q8`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v1 · `sha256:608707987dfc8f13…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 620  ·  **JSON-valid:** 90.5% (59 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 620 | 26.4% [25.5, 27.3] | 20.3% | 0% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 200 | 27.6% [26.0, 29.2] | 23.3% | 0% |
| unseen_station | dissent | 60 | 26.9% [23.6, 29.7] | 39.6% | 0% |
| unseen_time | clean | 200 | 27.1% [25.5, 28.6] | 21.0% | 0% |
| unseen_time | dissent | 60 | 33.0% [30.9, 35.2] | 29.6% | 0% |
| unseen_time | parse_fail | 100 | 18.0% [16.4, 19.5] | 5.7% | 0% |

## Reproducibility
Every number above re-derives from:
- eval set `v1` — sha256 `608707987dfc8f139775021b48895fbfacba1c17c77e4bb733b1f0b61bd95aba`
- prompt `decode-json-v1`  ·  model `gemma-3-1b-it-Q8`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
