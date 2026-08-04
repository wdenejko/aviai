# Run report — gemma-4-e4b-v2-r16-metar1500 (smoke 1500)

- **Date:** 2026-08-04T09:20:03Z
- **Model:** `gemma-4-e4b-v2-r16-metar1500 (smoke 1500)`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v2 · `sha256:6e19b0eb5d7a1dd6…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 1500  ·  **JSON-valid:** 100.0% (0 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 1500 | 99.5% [99.3, 99.6] | 6.1% | 91% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 519 | 99.3% [99.0, 99.5] | 0.3% | 92% |
| unseen_station | dissent | 133 | 99.5% [99.1, 99.8] | 0.0% | 95% |
| unseen_time | clean | 475 | 99.5% [99.3, 99.7] | 0.2% | 96% |
| unseen_time | dissent | 145 | 99.3% [98.8, 99.7] | 0.0% | 94% |
| unseen_time | parse_fail | 228 | 99.8% [99.6, 100.0] | 28.2% | 74% |

## Reproducibility
Every number above re-derives from:
- eval set `v2` — sha256 `6e19b0eb5d7a1dd674e2b105179efeb422f085fe15a3ddd57ddea600ffd9ee39`
- prompt `decode-json-v1`  ·  model `gemma-4-e4b-v2-r16-metar1500 (smoke 1500)`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
