# Run report — gemma-4-E4B-it-Q8-v2

- **Date:** 2026-07-31T03:58:40Z
- **Model:** `gemma-4-E4B-it-Q8-v2`  ·  **Prompt:** `decode-json-v1`
- **Eval:** v2 · `sha256:6e19b0eb5d7a1dd6…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 256}`
- **Records:** 6200  ·  **JSON-valid:** 100.0% (0 unusable)

Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI; **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 6200 | 88.8% [88.5, 89.1] | 21.4% | 25% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 2000 | 89.9% [89.4, 90.4] | 16.8% | 30% |
| unseen_station | dissent | 600 | 85.8% [84.8, 86.7] | 18.2% | 21% |
| unseen_time | clean | 2000 | 89.8% [89.4, 90.3] | 17.2% | 25% |
| unseen_time | dissent | 600 | 86.8% [86.0, 87.6] | 3.4% | 19% |
| unseen_time | parse_fail | 1000 | 87.5% [86.8, 88.2] | 40.4% | 20% |

## Reproducibility
Every number above re-derives from:
- eval set `v2` — sha256 `6e19b0eb5d7a1dd674e2b105179efeb422f085fe15a3ddd57ddea600ffd9ee39`
- prompt `decode-json-v1`  ·  model `gemma-4-E4B-it-Q8-v2`  ·  decode `{'temperature': 0.0, 'max_tokens': 256}`

Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).
