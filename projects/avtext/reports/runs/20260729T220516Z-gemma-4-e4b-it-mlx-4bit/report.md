# Run report — gemma-4-E4B-it-MLX-4bit

- **Date:** 2026-07-29T22:05:16Z
- **Model:** `gemma-4-E4B-it-MLX-4bit`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v1 · `sha256:608707987dfc8f13…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 620  ·  **JSON-valid:** 100.0% (0 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 620 | 77.8% [76.8, 78.9] | 1.5% | 4% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 200 | 80.4% [78.3, 82.4] | 1.0% | 6% |
| unseen_station | dissent | 60 | 70.0% [66.7, 73.2] | 0.0% | 0% |
| unseen_time | clean | 200 | 79.4% [77.6, 81.2] | 2.9% | 7% |
| unseen_time | dissent | 60 | 79.7% [76.0, 83.0] | 0.0% | 2% |
| unseen_time | parse_fail | 100 | 73.0% [71.5, 74.5] | 1.3% | 0% |

## Reproducibility
Every number above re-derives from:
- eval set `v1` — sha256 `608707987dfc8f139775021b48895fbfacba1c17c77e4bb733b1f0b61bd95aba`
- prompt `decode-json-v1`  ·  model `gemma-4-E4B-it-MLX-4bit`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
