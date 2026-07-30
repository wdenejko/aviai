# Run report — gemma-4-E4B-it-Q8-lora

- **Date:** 2026-07-30T15:42:09Z
- **Model:** `gemma-4-E4B-it-Q8-lora`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v1 · `sha256:608707987dfc8f13…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 620  ·  **JSON-valid:** 100.0% (0 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 620 | 98.3% [98.0, 98.6] | 0.4% | 84% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 200 | 98.1% [97.6, 98.6] | 0.0% | 82% |
| unseen_station | dissent | 60 | 97.2% [96.1, 98.3] | 0.0% | 72% |
| unseen_time | clean | 200 | 98.4% [97.8, 98.9] | 0.8% | 84% |
| unseen_time | dissent | 60 | 98.3% [97.4, 99.2] | 0.0% | 83% |
| unseen_time | parse_fail | 100 | 99.4% [98.7, 99.9] | 0.6% | 94% |

## Reproducibility
Every number above re-derives from:
- eval set `v1` — sha256 `608707987dfc8f139775021b48895fbfacba1c17c77e4bb733b1f0b61bd95aba`
- prompt `decode-json-v1`  ·  model `gemma-4-E4B-it-Q8-lora`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
