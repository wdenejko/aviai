# Run report — gemma-4-E4B-it-Q8-v2-r16

- **Date:** 2026-07-31T17:03:22Z
- **Model:** `gemma-4-E4B-it-Q8-v2-r16`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v2 · `sha256:6e19b0eb5d7a1dd6…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 6200  ·  **JSON-valid:** 100.0% (0 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 6200 | 99.4% [99.4, 99.5] | 7.2% | 90% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 2000 | 99.4% [99.3, 99.5] | 0.2% | 94% |
| unseen_station | dissent | 600 | 99.5% [99.3, 99.7] | 0.0% | 95% |
| unseen_time | clean | 2000 | 99.5% [99.4, 99.6] | 0.2% | 95% |
| unseen_time | dissent | 600 | 99.2% [99.0, 99.4] | 0.0% | 92% |
| unseen_time | parse_fail | 1000 | 99.6% [99.5, 99.8] | 31.3% | 70% |

## Reproducibility
Every number above re-derives from:
- eval set `v2` — sha256 `6e19b0eb5d7a1dd674e2b105179efeb422f085fe15a3ddd57ddea600ffd9ee39`
- prompt `decode-json-v1`  ·  model `gemma-4-E4B-it-Q8-v2-r16`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
