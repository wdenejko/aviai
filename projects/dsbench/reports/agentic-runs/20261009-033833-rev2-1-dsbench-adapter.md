# dsbench agentic run (pi harness): rev2-1-dsbench-adapter

- harness: `pi` (thinking high)  model: `qwen36`  provider: `dashi-qwen36`  k=5  reply cap: 12288 tokens
- when: 2026-10-09T03:38:33
- overall: **96/115** passing runs  ·  **19/23** problems pass by majority

| id | category | difficulty | passes | statuses | median latency | note |
|---|---|---|---|---|---|---|
| da_all_flights_avg_delay | da | medium | 3/5 | ok×3, wrong×2 | 10.3s | expected ~22.14 min; got 22.2 (raw '22.2') |
| da_cancel_dow | da | medium | 2/5 | wrong×3, ok×2 | 6.9s | expected Thursday (cancel rate by weekday: Thursday=9.36%, Wednesday=8 |
| da_cancel_weather_share | da | medium | 5/5 | ok×5 | 5.0s |  |
| da_delay_attribution | da | hard | 5/5 | ok×5 | 5.3s |  |
| da_delay_deviation | da | hard | 5/5 | ok×5 | 42.8s |  |
| da_delay_streak | da | hard | 5/5 | ok×5 | 58.8s |  |
| da_hub_delay | da | medium | 5/5 | ok×5 | 4.5s |  |
| da_redeye_count | da | medium | 3/5 | ok×3, wrong×2 | 7.0s | expected 6538 red-eye departures; got 590 (raw '590') |
| da_ts_delay | da | hard | 4/5 | ok×4, timeout×1 | 10.6s | pi exceeded 600s |
| da_utc_peak_hour | da | hard | 2/5 | wrong×3, ok×2 | 52.1s | expected UTC hour 13; got 5 (raw '5') [1 tool error(s) during run] |
| da_weekend_delay | da | medium | 2/5 | wrong×3, ok×2 | 10.3s | expected weekend/weekday ~24.3/21.5; got 24.5/20.9 |
| da_weighted_ontime | da | medium | 5/5 | ok×5 | 4.9s |  |
| da_worst_dep_hour | da | medium | 5/5 | ok×5 | 9.5s |  |
| de_carrier_ontime | de | hard | 2/5 | wrong×3, ok×2 | 37.0s | ontime_rate values do not match the data |
| de_hub_daily | de | hard | 4/5 | ok×4, wrong×1 | 35.5s | value mismatches — n_dep:122 n_metar:0 avg_dep_delay:23 |
| de_recovery_leaderboard | de | hard | 5/5 | ok×5 | 43.9s |  |
| de_route_leaderboard | de | hard | 5/5 | ok×5 | 30.2s |  |
| de_taf_latest | de | hard | 5/5 | ok×5 | 26.5s |  |
| de_taf_summary | de | medium | 5/5 | ok×5 | 12.5s |  |
| ds_cancel_predict | ds | hard | 5/5 | ok×5 | 88.5s |  |
| ds_delay_predict | ds | hard | 5/5 | ok×5 | 70.5s |  |
| ds_notam_classify | ds | hard | 5/5 | ok×5 | 62.2s |  |
| ds_taxi_regression | ds | hard | 4/5 | ok×4, timeout×1 | 217.8s | pi exceeded 600s |
