# dsbench agentic run (pi harness): ornith-pi-baseline-thinkon

- harness: `pi` (thinking high)  model: `ornith`  provider: `dashi-ornith`
- when: 2026-09-18T10:50:12
- score: **4/6** (67%)

| id | category | difficulty | status | steps | tool calls | latency | note |
|---|---|---|---|---|---|---|---|
| da_cancel_dow | da | medium | wrong | 2 | 1 | 5.3s | expected Thursday (cancel rate by weekday: Thursday=9.36%, Wednesday=8.52%, Monday=2.13%,  |
| da_hub_delay | da | medium | ok | 2 | 1 | 4.3s |  |
| da_ts_delay | da | hard | ok | 10 | 10 | 60.1s |  |
| de_hub_daily | de | hard | ok | 11 | 17 | 78.3s |  |
| de_taf_summary | de | medium | ok | 6 | 7 | 19.1s |  |
| ds_notam_classify | ds | hard | wrong | 10 | 12 | 41.8s | could not read prob_ds_notam_classify.notam_pred (id, predicted): Received ClickHouse exce |
