# Run report — gemma-3n-E4B-it-Q8

- **Date:** 2026-07-30T08:44:58Z
- **Model:** `gemma-3n-E4B-it-Q8`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v1 · `sha256:608707987dfc8f13…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 620  ·  **JSON-valid:** 99.2% (5 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 620 | 68.8% [67.6, 70.1] | 8.5% | 4% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 200 | 68.1% [65.7, 70.3] | 7.3% | 4% |
| unseen_station | dissent | 60 | 59.6% [56.7, 62.3] | 3.8% | 0% |
| unseen_time | clean | 200 | 68.8% [66.6, 70.9] | 10.5% | 8% |
| unseen_time | dissent | 60 | 66.3% [63.2, 69.7] | 5.6% | 0% |
| unseen_time | parse_fail | 100 | 77.8% [74.5, 80.7] | 9.4% | 0% |

## Reproducibility
Every number above re-derives from:
- eval set `v1` — sha256 `608707987dfc8f139775021b48895fbfacba1c17c77e4bb733b1f0b61bd95aba`
- prompt `decode-json-v1`  ·  model `gemma-3n-E4B-it-Q8`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
