# dsbench run: expert-ornith-nothink

- model: `ornith-1.5-q8`  endpoint: `http://localhost:18080`
- when: 2026-09-15T17:58:12 temp: 0.0 pass@1 thinking: off
- score: **2/3** (66.7%)

| category | easy | medium | hard | expert | total |
|---|---|---|---|---|---|
| data engineering | - | - | - | 1/1 | 1/1 |
| data analysis | - | - | - | 0/1 | 0/1 |
| data science | - | - | - | 1/1 | 1/1 |

## failures

| id | difficulty | status | reason |
|---|---|---|---|
| da_expert_01 | expert | error | ValueError: Can only compare identically-labeled Series objects |
