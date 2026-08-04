# TAF run report — gemma-4-e4b-taf-r16 (smoke 1500)

- **Date:** 2026-08-02T21:48:08Z
- **Model:** `gemma-4-e4b-taf-r16 (smoke 1500)`  ·  **Prompt:** `decode-taf-json-v1`
- **Eval:** taf-v1 · `sha256:f103708d438a686f…`
- **Decode:** `{"temperature": 0.0, "max_tokens": 1024}`
- **Records:** 1500  ·  **JSON-valid:** 97.1% (44 unusable)  ·  **period-count acc:** 95.7%

Columns: **value acc** = recall (hits / values that existed), 95% bootstrap CI; **halluc** = fabricated / truly-absent; **EM** = whole-forecast exact match.

## Overall
| scope | n | value acc [95% CI] | halluc | EM |
|---|--:|--|--:|--:|
| all | 1500 | 93.1% [91.2, 95.0] | 1.5% | 89% |

## By split × label
| split | label | n | value acc [95% CI] | halluc | EM |
|---|---|--:|--|--:|--:|
| unseen_station | clean | 718 | 93.9% [91.0, 96.5] | 0.1% | 96% |
| unseen_station | dissent | 21 | 90.6% [75.6, 100.0] | 7.7% | 0% |
| unseen_time | clean | 688 | 92.1% [89.0, 95.1] | 0.0% | 94% |
| unseen_time | dissent | 54 | 95.5% [88.8, 99.5] | 15.6% | 0% |
| unseen_time | parse_fail | 19 | 90.8% [89.2, 92.5] | 42.4% | 0% |

## Reproducibility
- eval `taf-v1` — sha256 `f103708d438a686ff8f187fc483e486c0f92027fe413a7576dc1753e561f2b22`
- prompt `decode-taf-json-v1`  ·  model `gemma-4-e4b-taf-r16 (smoke 1500)`  ·  decode `{'temperature': 0.0, 'max_tokens': 1024}`
