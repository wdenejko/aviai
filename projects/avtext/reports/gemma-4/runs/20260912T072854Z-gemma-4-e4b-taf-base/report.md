# TAF run report — gemma-4-e4b-taf-base

- **Date:** 2026-09-12T07:28:54Z
- **Model:** `gemma-4-e4b-taf-base`  ·  **Prompt:** `decode-taf-json-v1`
- **Eval:** taf-v1 · `sha256:f103708d438a686f…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 2048}`
- **Records:** 5294  ·  **JSON-valid:** 98.9% (58 unusable)  ·  **period-count acc:** 91.2%

Columns: **value acc** = recall (hits / values that existed), 95% bootstrap CI; **halluc** = fabricated / truly-absent; **EM** = whole-forecast exact match.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 5294 | 89.4% [88.9, 89.9] | 11.8% | 7% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 2500 | 89.9% [89.1, 90.6] | 10.2% | 10% |
| unseen_station | dissent | 53 | 87.6% [83.5, 91.3] | 15.0% | 0% |
| unseen_time | clean | 2500 | 89.6% [89.0, 90.1] | 10.9% | 5% |
| unseen_time | dissent | 162 | 84.4% [79.9, 88.3] | 20.0% | 0% |
| unseen_time | parse_fail | 79 | 80.6% [78.4, 82.6] | 48.6% | 0% |

## Reproducibility
- eval `taf-v1` — sha256 `f103708d438a686ff8f187fc483e486c0f92027fe413a7576dc1753e561f2b22`
- prompt `decode-taf-json-v1`  ·  model `gemma-4-e4b-taf-base`  ·  decode `{'temperature': 0.0, 'max_tokens': 2048}`
