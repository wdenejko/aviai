# Run report — gemma-4-E4B-it-Q8-v2-r64

- **Date:** 2026-07-31T10:31:33Z
- **Model:** `gemma-4-E4B-it-Q8-v2-r64`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v2 · `sha256:6e19b0eb5d7a1dd6…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 6200  ·  **JSON-valid:** 100.0% (0 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 6200 | 99.6% [99.6, 99.7] | 7.0% | 92% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 2000 | 99.5% [99.4, 99.6] | 0.3% | 95% |
| unseen_station | dissent | 600 | 99.8% [99.7, 99.9] | 0.1% | 98% |
| unseen_time | clean | 2000 | 99.6% [99.5, 99.7] | 0.1% | 96% |
| unseen_time | dissent | 600 | 99.3% [99.1, 99.5] | 0.0% | 94% |
| unseen_time | parse_fail | 1000 | 99.8% [99.7, 99.9] | 30.3% | 73% |

## Reproducibility
Every number above re-derives from:
- eval set `v2` — sha256 `6e19b0eb5d7a1dd674e2b105179efeb422f085fe15a3ddd57ddea600ffd9ee39`
- prompt `decode-json-v1`  ·  model `gemma-4-E4B-it-Q8-v2-r64`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
