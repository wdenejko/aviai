# TAF run report — gemma-4-e4b-taf-base (smoke 1500)

- **Date:** 2026-08-03T03:57:22Z
- **Model:** `gemma-4-e4b-taf-base (smoke 1500)`  ·  **Prompt:** `decode-taf-json-v1`
- **Eval:** taf-v1 · `sha256:f103708d438a686f…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 1024}`
- **Records:** 1500  ·  **JSON-valid:** 96.1% (59 unusable)  ·  **period-count acc:** 87.4%

Columns: **value acc** = recall (hits / values that existed), 95% bootstrap CI; **halluc** = fabricated / truly-absent; **EM** = whole-forecast exact match.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 1500 | 83.6% [81.8, 85.4] | 11.9% | 7% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 718 | 84.4% [81.5, 87.0] | 10.6% | 9% |
| unseen_station | dissent | 21 | 85.9% [79.1, 92.1] | 17.6% | 0% |
| unseen_time | clean | 688 | 83.1% [80.2, 85.8] | 11.1% | 6% |
| unseen_time | dissent | 54 | 80.3% [71.9, 87.5] | 20.0% | 0% |
| unseen_time | parse_fail | 19 | 81.7% [77.9, 85.2] | 44.8% | 0% |

## Reproducibility
- eval `taf-v1` — sha256 `f103708d438a686ff8f187fc483e486c0f92027fe413a7576dc1753e561f2b22`
- prompt `decode-taf-json-v1`  ·  model `gemma-4-e4b-taf-base (smoke 1500)`  ·  decode `{'temperature': 0.0, 'max_tokens': 1024}`
