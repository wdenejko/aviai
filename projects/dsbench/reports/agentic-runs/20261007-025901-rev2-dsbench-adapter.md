# dsbench agentic run (pi harness): rev2-dsbench-adapter

- harness: `pi` (thinking high)  model: `qwen36`  provider: `dashi-qwen36`  k=5  reply cap: 12288 tokens
- when: 2026-10-07T02:59:01
- overall: **103/115** passing runs  ·  **21/23** problems pass by majority

| id | category | difficulty | passes | statuses | median latency | note |
|---|---|---|---|---|---|---|
| da_all_flights_avg_delay | da | medium | 5/5 | ok×5 | 7.2s |  |
| da_cancel_dow | da | medium | 5/5 | ok×5 | 11.0s |  |
| da_cancel_weather_share | da | medium | 5/5 | ok×5 | 5.4s |  |
| da_delay_attribution | da | hard | 5/5 | ok×5 | 5.2s |  |
| da_delay_deviation | da | hard | 4/5 | ok×4, wrong×1 | 123.2s | expected DFW 2026-06-19; got iata=ATL day=2026-06-14 |
| da_delay_streak | da | hard | 5/5 | ok×5 | 50.2s |  |
| da_hub_delay | da | medium | 5/5 | ok×5 | 4.8s |  |
| da_redeye_count | da | medium | 2/5 | wrong×3, ok×2 | 13.1s | expected 6538 red-eye departures; got 10033 (raw '10033') |
| da_ts_delay | da | hard | 5/5 | ok×5 | 15.7s |  |
| da_utc_peak_hour | da | hard | 4/5 | ok×4, wrong×1 | 45.7s | expected UTC hour 13; got 5 (raw '5') |
| da_weekend_delay | da | medium | 1/5 | wrong×4, ok×1 | 8.3s | expected weekend/weekday ~24.3/21.5; got 22.2/21.0 |
| da_weighted_ontime | da | medium | 5/5 | ok×5 | 4.8s |  |
| da_worst_dep_hour | da | medium | 5/5 | ok×5 | 10.5s |  |
| de_carrier_ontime | de | hard | 5/5 | ok×5 | 75.9s |  |
| de_hub_daily | de | hard | 4/5 | ok×4, wrong×1 | 42.4s | value mismatches — n_dep:122 n_metar:0 avg_dep_delay:23 |
| de_recovery_leaderboard | de | hard | 5/5 | ok×5 | 40.9s |  |
| de_route_leaderboard | de | hard | 5/5 | ok×5 | 36.8s |  |
| de_taf_latest | de | hard | 5/5 | ok×5 | 40.9s |  |
| de_taf_summary | de | medium | 5/5 | ok×5 | 17.7s |  |
| ds_cancel_predict | ds | hard | 5/5 | ok×5 | 54.0s |  |
| ds_delay_predict | ds | hard | 5/5 | ok×5 | 192.0s |  |
| ds_notam_classify | ds | hard | 4/5 | ok×4, wrong×1 | 59.2s | 3606 of 4241 test rows have no prediction [3 tool error(s) during run] |
| ds_taxi_regression | ds | hard | 4/5 | ok×4, timeout×1 | 203.7s | pi exceeded 600s |
