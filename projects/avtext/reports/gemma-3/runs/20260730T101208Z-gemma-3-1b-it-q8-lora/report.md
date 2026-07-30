# Run report — gemma-3-1b-it-Q8-lora

- **Date:** 2026-07-30T10:12:08Z
- **Model:** `gemma-3-1b-it-Q8-lora`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v1 · `sha256:608707987dfc8f13…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 620  ·  **JSON-valid:** 100.0% (0 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 620 | 92.9% [92.4, 93.3] | 0.3% | 36% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 200 | 92.7% [91.9, 93.4] | 0.0% | 32% |
| unseen_station | dissent | 60 | 89.0% [87.6, 90.3] | 0.0% | 10% |
| unseen_time | clean | 200 | 93.0% [92.1, 93.8] | 0.4% | 38% |
| unseen_time | dissent | 60 | 92.1% [91.1, 93.2] | 0.0% | 22% |
| unseen_time | parse_fail | 100 | 96.1% [94.9, 97.2] | 0.6% | 68% |

## Reproducibility
Every number above re-derives from:
- eval set `v1` — sha256 `608707987dfc8f139775021b48895fbfacba1c17c77e4bb733b1f0b61bd95aba`
- prompt `decode-json-v1`  ·  model `gemma-3-1b-it-Q8-lora`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
