# dsbench agentic run (pi harness): ornith-pi-baseline-v2

- harness: `pi` (thinking high)  model: `ornith`  provider: `dashi-ornith`
- when: 2026-09-18T11:02:40
- score: **4/6** (67%)

| id | category | difficulty | status | steps | tool calls | latency | note |
|---|---|---|---|---|---|---|---|
| da_cancel_dow | da | medium | wrong | 3 | 2 | 9.2s | expected Thursday (cancel rate by weekday: Thursday=9.36%, Wednesday=8.52%, Monday=2.13%,  |
| da_hub_delay | da | medium | ok | 2 | 1 | 4.5s |  |
| da_ts_delay | da | hard | ok | 14 | 15 | 79.6s |  |
| de_hub_daily | de | hard | wrong | 10 | 11 | 97.5s | hub_daily is empty |
| de_taf_summary | de | medium | ok | 10 | 10 | 35.7s |  |
| ds_notam_classify | ds | hard | ok | 12 | 16 | 76.8s |  |
