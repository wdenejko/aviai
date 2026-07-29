# Run report — gemma-4-E4B-it-MLX-4bit

- **Date:** 2026-07-29T17:03:54Z
- **Model:** `gemma-4-E4B-it-MLX-4bit`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v1 · `sha256:608707987dfc8f13…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 620  ·  **JSON-valid:** 96.0% (25 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 620 | 75.1% [73.5, 76.7] | 1.5% | 4% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 200 | 78.5% [75.8, 81.1] | 1.0% | 6% |
| unseen_station | dissent | 60 | 67.4% [62.3, 72.0] | 0.0% | 0% |
| unseen_time | clean | 200 | 76.1% [73.1, 78.9] | 2.9% | 7% |
| unseen_time | dissent | 60 | 74.1% [68.0, 79.7] | 0.0% | 2% |
| unseen_time | parse_fail | 100 | 71.5% [68.8, 73.8] | 1.3% | 0% |

## Reproducibility
Every number above re-derives from:
- eval set `v1` — sha256 `608707987dfc8f139775021b48895fbfacba1c17c77e4bb733b1f0b61bd95aba`
- prompt `decode-json-v1`  ·  model `gemma-4-E4B-it-MLX-4bit`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
