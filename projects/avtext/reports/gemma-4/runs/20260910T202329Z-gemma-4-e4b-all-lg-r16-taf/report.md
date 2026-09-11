# TAF run report — gemma-4-e4b-all-lg-r16-taf

- **Date:** 2026-09-10T20:23:29Z
- **Model:** `gemma-4-e4b-all-lg-r16-taf`  ·  **Prompt:** `decode-taf-json-v1`
- **Eval:** taf-v1 · `sha256:f103708d438a686f…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 2048}`
- **Records:** 5294  ·  **JSON-valid:** 99.9% (3 unusable)  ·  **period-count acc:** 98.8%

Columns: **value acc** = recall (hits / values that existed), 95% bootstrap CI; **halluc** = fabricated / truly-absent; **EM** = whole-forecast exact match.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 5294 | 99.6% [99.4, 99.8] | 1.7% | 93% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 2500 | 99.7% [99.2, 100.0] | 0.0% | 99% |
| unseen_station | dissent | 53 | 99.6% [99.1, 100.0] | 7.5% | 0% |
| unseen_time | clean | 2500 | 99.9% [99.8, 100.0] | 0.0% | 99% |
| unseen_time | dissent | 162 | 98.4% [97.2, 99.3] | 12.8% | 1% |
| unseen_time | parse_fail | 79 | 89.0% [87.3, 90.4] | 46.5% | 0% |

## Reproducibility
- eval `taf-v1` — sha256 `f103708d438a686ff8f187fc483e486c0f92027fe413a7576dc1753e561f2b22`
- prompt `decode-taf-json-v1`  ·  model `gemma-4-e4b-all-lg-r16-taf`  ·  decode `{'temperature': 0.0, 'max_tokens': 2048}`
