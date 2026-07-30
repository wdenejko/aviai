# Run report — qwen2.5-0.5b-it-Q8

- **Date:** 2026-07-30T07:16:22Z
- **Model:** `qwen2.5-0.5b-it-Q8`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v1 · `sha256:608707987dfc8f13…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 620  ·  **JSON-valid:** 98.7% (8 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 620 | 5.3% [4.8, 5.7] | 1.0% | 0% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 200 | 4.4% [3.6, 5.3] | 1.5% | 0% |
| unseen_station | dissent | 60 | 4.8% [3.3, 6.3] | 0.0% | 0% |
| unseen_time | clean | 200 | 4.9% [4.1, 5.8] | 0.8% | 0% |
| unseen_time | dissent | 60 | 2.6% [1.6, 3.8] | 0.0% | 0% |
| unseen_time | parse_fail | 100 | 9.8% [9.2, 10.3] | 1.3% | 0% |

## Reproducibility
Every number above re-derives from:
- eval set `v1` — sha256 `608707987dfc8f139775021b48895fbfacba1c17c77e4bb733b1f0b61bd95aba`
- prompt `decode-json-v1`  ·  model `qwen2.5-0.5b-it-Q8`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
