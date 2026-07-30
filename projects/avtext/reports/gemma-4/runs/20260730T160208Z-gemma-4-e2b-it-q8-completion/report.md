# Run report — gemma-4-E2B-it-Q8-completion

- **Date:** 2026-07-30T16:02:08Z
- **Model:** `gemma-4-E2B-it-Q8-completion`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v1 · `sha256:608707987dfc8f13…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 620  ·  **JSON-valid:** 98.2% (11 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 620 | 70.0% [68.7, 71.2] | 30.1% | 3% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 200 | 66.2% [63.6, 68.9] | 28.6% | 2% |
| unseen_station | dissent | 60 | 63.9% [60.6, 67.0] | 35.8% | 0% |
| unseen_time | clean | 200 | 72.0% [70.1, 73.9] | 32.4% | 6% |
| unseen_time | dissent | 60 | 70.5% [67.2, 73.6] | 46.3% | 0% |
| unseen_time | parse_fail | 100 | 77.3% [75.6, 78.9] | 21.4% | 0% |

## Reproducibility
Every number above re-derives from:
- eval set `v1` — sha256 `608707987dfc8f139775021b48895fbfacba1c17c77e4bb733b1f0b61bd95aba`
- prompt `decode-json-v1`  ·  model `gemma-4-E2B-it-Q8-completion`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
