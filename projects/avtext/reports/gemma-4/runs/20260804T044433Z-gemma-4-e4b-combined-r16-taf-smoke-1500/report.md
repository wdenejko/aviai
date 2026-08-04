# TAF run report — gemma-4-e4b-combined-r16-taf (smoke 1500)

- **Date:** 2026-08-04T04:44:33Z
- **Model:** `gemma-4-e4b-combined-r16-taf (smoke 1500)`  ·  **Prompt:** `decode-taf-json-v1`
- **Eval:** taf-v1 · `sha256:f103708d438a686f…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 1024}`
- **Records:** 1500  ·  **JSON-valid:** 97.1% (44 unusable)  ·  **period-count acc:** 95.9%

Columns: **value acc** = recall (hits / values that existed), 95% bootstrap CI; **halluc** = fabricated / truly-absent; **EM** = whole-forecast exact match.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 1500 | 93.2% [91.3, 95.1] | 1.5% | 90% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 718 | 93.9% [91.0, 96.5] | 0.0% | 96% |
| unseen_station | dissent | 21 | 90.6% [75.6, 100.0] | 8.1% | 0% |
| unseen_time | clean | 688 | 92.2% [89.1, 95.1] | 0.1% | 96% |
| unseen_time | dissent | 54 | 96.3% [90.0, 99.8] | 14.3% | 0% |
| unseen_time | parse_fail | 19 | 90.4% [88.8, 91.9] | 42.4% | 0% |

## Reproducibility
- eval `taf-v1` — sha256 `f103708d438a686ff8f187fc483e486c0f92027fe413a7576dc1753e561f2b22`
- prompt `decode-taf-json-v1`  ·  model `gemma-4-e4b-combined-r16-taf (smoke 1500)`  ·  decode `{'temperature': 0.0, 'max_tokens': 1024}`
