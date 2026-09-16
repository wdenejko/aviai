# dsbench run: base-ornith-nothink

- model: `ornith-1.5-q8`  endpoint: `http://localhost:18080`
- when: 2026-09-15T17:59:59 temp: 0.0 pass@1 thinking: off
- score: **16/21** (76.2%)

| category | easy | medium | hard | expert | total |
|---|---|---|---|---|---|
| data engineering | 2/2 | 1/2 | 1/2 | 1/1 | 5/7 |
| data analysis | 2/2 | 2/2 | 1/2 | 0/1 | 5/7 |
| data science | 2/2 | 2/2 | 1/2 | 1/1 | 6/7 |

## failures

| id | difficulty | status | reason |
|---|---|---|---|
| da_expert_01 | expert | error | ValueError: Can only compare identically-labeled Series objects |
| da_hard_01 | hard | wrong | incorrect result |
| de_hard_02 | hard | wrong | incorrect result |
| de_medium_02 | medium | wrong | incorrect result |
| ds_hard_02 | hard | error | NameError: name 'argsort' is not defined |
