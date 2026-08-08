# TAF run report — gemma-4-e4b-all-r16-taf (smoke 1500)

- **Date:** 2026-08-08T00:43:04Z
- **Model:** `gemma-4-e4b-all-r16-taf (smoke 1500)`  ·  **Prompt:** `decode-taf-json-v1`
- **Eval:** taf-v1 · `sha256:f103708d438a686f…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 1024}`
- **Records:** 1500  ·  **JSON-valid:** 96.9% (46 unusable)  ·  **period-count acc:** 95.9%

Columns: **value acc** = recall (hits / values that existed), 95% bootstrap CI; **halluc** = fabricated / truly-absent; **EM** = whole-forecast exact match.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 1500 | 93.1% [91.2, 95.0] | 1.4% | 90% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 718 | 94.1% [91.2, 96.7] | 0.1% | 97% |
| unseen_station | dissent | 21 | 92.6% [78.1, 100.0] | 5.8% | 0% |
| unseen_time | clean | 688 | 91.7% [88.5, 94.5] | 0.0% | 95% |
| unseen_time | dissent | 54 | 97.0% [93.3, 99.6] | 14.0% | 0% |
| unseen_time | parse_fail | 19 | 91.5% [89.8, 93.2] | 41.9% | 0% |

## Reproducibility
- eval `taf-v1` — sha256 `f103708d438a686ff8f187fc483e486c0f92027fe413a7576dc1753e561f2b22`
- prompt `decode-taf-json-v1`  ·  model `gemma-4-e4b-all-r16-taf (smoke 1500)`  ·  decode `{'temperature': 0.0, 'max_tokens': 1024}`
