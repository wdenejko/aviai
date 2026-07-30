# Run report — gemma-3-270m-it-Q8

- **Date:** 2026-07-30T07:11:37Z
- **Model:** `gemma-3-270m-it-Q8`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v1 · `sha256:608707987dfc8f13…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 620  ·  **JSON-valid:** 69.2% (191 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 620 | 17.5% [16.4, 18.6] | 66.9% | 0% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 200 | 16.9% [14.8, 18.7] | 66.5% | 0% |
| unseen_station | dissent | 60 | 14.5% [10.9, 18.1] | 56.6% | 0% |
| unseen_time | clean | 200 | 17.9% [15.9, 20.1] | 61.3% | 0% |
| unseen_time | dissent | 60 | 13.5% [10.4, 16.6] | 53.7% | 0% |
| unseen_time | parse_fail | 100 | 22.2% [19.6, 24.9] | 83.6% | 0% |

## Reproducibility
Every number above re-derives from:
- eval set `v1` — sha256 `608707987dfc8f139775021b48895fbfacba1c17c77e4bb733b1f0b61bd95aba`
- prompt `decode-json-v1`  ·  model `gemma-3-270m-it-Q8`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
