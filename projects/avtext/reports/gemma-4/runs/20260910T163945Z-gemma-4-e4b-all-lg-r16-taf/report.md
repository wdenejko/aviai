# TAF run report — gemma-4-e4b-all-lg-r16-taf

- **Date:** 2026-09-10T16:39:45Z
- **Model:** `gemma-4-e4b-all-lg-r16-taf`  ·  **Prompt:** `decode-taf-json-v1`
- **Eval:** taf-v1 · `sha256:f103708d438a686f…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 1024}`
- **Records:** 5294  ·  **JSON-valid:** 97.1% (152 unusable)  ·  **period-count acc:** 96.1%

Columns: **value acc** = recall (hits / values that existed), 95% bootstrap CI; **halluc** = fabricated / truly-absent; **EM** = whole-forecast exact match.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 5294 | 93.3% [92.3, 94.4] | 1.6% | 91% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 2500 | 94.2% [92.8, 95.6] | 0.0% | 97% |
| unseen_station | dissent | 53 | 93.3% [84.3, 99.8] | 7.3% | 0% |
| unseen_time | clean | 2500 | 93.3% [91.8, 94.7] | 0.0% | 96% |
| unseen_time | dissent | 162 | 85.1% [77.4, 92.3] | 12.4% | 1% |
| unseen_time | parse_fail | 79 | 89.0% [87.3, 90.4] | 45.1% | 1% |

## Reproducibility
- eval `taf-v1` — sha256 `f103708d438a686ff8f187fc483e486c0f92027fe413a7576dc1753e561f2b22`
- prompt `decode-taf-json-v1`  ·  model `gemma-4-e4b-all-lg-r16-taf`  ·  decode `{'temperature': 0.0, 'max_tokens': 1024}`
