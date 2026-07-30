# Run report — gemma-4-E2B-it-Q8-lora

- **Date:** 2026-07-30T13:05:53Z
- **Model:** `gemma-4-E2B-it-Q8-lora`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v1 · `sha256:608707987dfc8f13…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 620  ·  **JSON-valid:** 100.0% (0 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 620 | 92.9% [92.4, 93.4] | 0.8% | 38% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 200 | 93.8% [93.0, 94.7] | 0.0% | 46% |
| unseen_station | dissent | 60 | 91.1% [89.2, 92.9] | 0.0% | 28% |
| unseen_time | clean | 200 | 93.4% [92.6, 94.3] | 2.1% | 42% |
| unseen_time | dissent | 60 | 93.4% [92.2, 94.7] | 0.0% | 35% |
| unseen_time | parse_fail | 100 | 90.9% [89.8, 91.9] | 0.6% | 22% |

## Reproducibility
Every number above re-derives from:
- eval set `v1` — sha256 `608707987dfc8f139775021b48895fbfacba1c17c77e4bb733b1f0b61bd95aba`
- prompt `decode-json-v1`  ·  model `gemma-4-E2B-it-Q8-lora`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
