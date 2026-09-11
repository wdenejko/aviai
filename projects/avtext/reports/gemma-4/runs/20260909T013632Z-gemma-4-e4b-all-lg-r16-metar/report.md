# Run report — gemma-4-e4b-all-lg-r16-metar

- **Date:** 2026-09-09T01:36:32Z
- **Model:** `gemma-4-e4b-all-lg-r16-metar`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v2 · `sha256:6e19b0eb5d7a1dd6…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 6200  ·  **JSON-valid:** 100.0% (0 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 6200 | 99.9% [99.8, 99.9] | 7.0% | 94% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 2000 | 99.8% [99.8, 99.9] | 0.1% | 98% |
| unseen_station | dissent | 600 | 99.8% [99.7, 99.9] | 0.0% | 98% |
| unseen_time | clean | 2000 | 100.0% [99.9, 100.0] | 0.0% | 100% |
| unseen_time | dissent | 600 | 99.7% [99.6, 99.9] | 0.0% | 98% |
| unseen_time | parse_fail | 1000 | 99.8% [99.7, 99.9] | 30.8% | 72% |

## Reproducibility
Every number above re-derives from:
- eval set `v2` — sha256 `6e19b0eb5d7a1dd674e2b105179efeb422f085fe15a3ddd57ddea600ffd9ee39`
- prompt `decode-json-v1`  ·  model `gemma-4-e4b-all-lg-r16-metar`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
