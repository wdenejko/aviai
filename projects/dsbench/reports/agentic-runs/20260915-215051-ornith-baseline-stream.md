# dsbench agentic run: ornith-baseline-stream

- model: `ornith`  endpoint: `http://localhost:18080`  thinking: off
- when: 2026-09-15T21:50:51
- score: **4/6** (67%)

| id | category | difficulty | status | tool calls | steps | latency | note |
|---|---|---|---|---|---|---|---|
| da_cancel_dow | da | medium | wrong | 3 | 3 | 7.1s | expected Thursday (cancel rate by weekday: Thursday=9.36%, Wednesday=8.52%, Monday=2.13%,  |
| da_hub_delay | da | medium | wrong | 2 | 2 | 3.5s | expected DFW (avg dep delay by hub: DFW=31.9, ORD=22.9, DEN=22.0, LAX=18.3, ATL=14.8); got |
| da_ts_delay | da | hard | ok | 4 | 4 | 12.4s |  |
| de_hub_daily | de | hard | ok | 7 | 6 | 22.1s |  |
| de_taf_summary | de | medium | ok | 6 | 6 | 16.6s |  |
| ds_notam_classify | ds | hard | ok | 12 | 10 | 59.8s |  |
