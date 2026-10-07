# The held-out probe and dsbench: Revision 2's adapter against the base, through pi

ADR-004 Revision 2, action item 10. Run 2026-10-06/07 on dashi, in one window of 3 hours 36
minutes. The base and the final adapter go through the owner's pi harness, k = 5 runs a problem,
with thinking on. The tooling is `patches/rev2_probe_mac.sh` and `agentic/compare_runs.py`. The
runs are in `reports/agentic-runs/`, and the comparisons are in `20261007-probe-dsbench.json`.

## Summary

- **The ClickHouse weekday convention carries over to prompts written nothing like the training
  rows:**
  - The held-out probe's weekday task (Tuesday visits in a clinic table) goes from 0 of 5 runs to
    4 of 5 (Fisher p = 0.048). The base writes `toDayOfWeek(visit_date) = 3`, MySQL's Tuesday,
    every time. The adapter writes `= 2` four times.
  - dsbench's `da_cancel_dow` (the weekday with the highest cancellation rate, on the aviation
    warehouse) goes from 0 of 5 to 5 of 5 (p = 0.008). The base names Wednesday because it
    translates `toDayOfWeek` with Sunday = 1.
- **The weekend half of it carries over only in part.** On dsbench's `da_weekend_delay`, the base
  writes `toDayOfWeek(...) IN (1, 7)` every time, MySQL's weekend.
  - The adapter writes `IN (6, 7)` twice. Once it was right in full; the other time it took the
    wrong population.
  - The other three times it wrote `IN (1, 6)`: ISO's Saturday next to the old Sunday. That is a
    new mistake, half of the convention. It passes 1 of 5.
- **Totals:**
  - Probe: 25 → 28 of 30 runs passed; 5 → 6 of 6 problems passed by majority.
  - dsbench: 94 → 103 of 115 runs; 19 → 21 of 23 problems by majority (3 gained, 1 lost;
    McNemar p = 0.63).
- **Nothing got significantly worse.** One problem lost its majority: `da_redeye_count` went from
  5 to 2 of 5 (p = 0.17). Its three misses are a reading of the population, not a convention.
  "Across the five hub airports ... departures" became flights from OR to the hubs twice, and
  flights between hubs once.
- **The timezone direction improves on dsbench, without significance:** `da_utc_peak_hour` 2 → 4
  of 5. On the probe, the base got all 5 (it got 3 on 2026-09-20) and the adapter 4; the adapter's
  miss subtracted the offset. Over the two suites, the adapter misses this trap 2 times in 10, the
  base 3.
- **The agentic ML problems hold:** the probe's two at 5 of 5 in both states. dsbench's four ds
  problems pass 19 → 18 of 20 runs: the adapter's misses are a run that left most test rows
  without a prediction and a timeout; the base's one miss was a timeout.

## What was run

- **The window:** `battery_window.sh` with one step, `hold:probe`. It served the base with the
  final adapter loaded (`2026-10-06-qwen36-rev2-gate-final/lora.gguf`, the one the gate and the
  Target A test ran). Box clock 21:24 to 01:00 (CEST 23:23 to 02:59). Production was off; OCR was
  restored at the end.
- **Parity, first:** scale 0 gave the bare base's greedy replies on all 8 prompts, and again after
  scale 1. Scale 1 changed 4 of them.
- **The Mac:** `rev2_probe_mac.sh` ran four steps: the probe (6 problems), then dsbench (23), each
  k = 5, first with the server's adapter scale at 0 and then at 1. The script set the scale and
  read it back before each step. It put the scale back to 0 and released the hold at the end.
  - Probe: base 7 minutes, adapter 10.
  - dsbench: base 91 minutes, adapter 105. The ds problems' model training dominates.
- **pi's settings, the owner's:** thinking on (`--thinking high` sends `enable_thinking: true`),
  temperature 0, 12,288 tokens a reply. Checked against a stand-in that recorded pi 0.84.4's
  request bodies. The probe's baseline of 2026-09-20 ran with the same settings, on the
  production server.
- **Grading:** each problem's own oracle, on the Mac's ClickHouse sandbox, with the agent on its
  own login (nothing withheld in its reach). Before the window, both oracle gates passed without a
  model: the probe 6 of 6, dsbench 23 of 23.

## The probe

| Problem | Skill | Base | Adapter | Fisher p | Adapter's misses |
|---|---|---:|---:|---:|---|
| probe_weekday_visits | dialect weekday | 0/5 | 4/5 | 0.048 | `toDayOfWeek(...) = 3` once |
| probe_utc_peak_trips | timezone direction | 5/5 | 4/5 | 1 | subtracted the offset once |
| probe_incident_share | population share | 5/5 | 5/5 | 1 | |
| probe_pooled_rating | pooled rating | 5/5 | 5/5 | 1 | |
| probe_loan_default | agentic ML (ROC-AUC) | 5/5 | 5/5 | 1 | |
| probe_energy_demand | agentic ML (MAE) | 5/5 | 5/5 | 1 | |

The probe is the generalisation test ADR-004 set: the same three skills on domains in neither
dsbench nor the training data, hand-written and run with a neutral system prompt. A dsbench gain
without a probe gain would have been the overfitting alarm. Both moved, and on the same skill.

## dsbench

The problems that moved, or that either state fails:

| Problem | Base | Adapter | Fisher p | What happened |
|---|---:|---:|---:|---|
| da_cancel_dow | 0/5 | 5/5 | 0.008 | the base translates weekday numbers with Sunday = 1; the adapter with ISO |
| de_carrier_ontime | 2/5 | 5/5 | 0.17 | the base's table was wrong 3 times: its ranks once, its on-time rates twice |
| da_utc_peak_hour | 2/5 | 4/5 | 0.52 | the direction trap: the base subtracted the offset 3 times, the adapter once |
| de_route_leaderboard | 4/5 | 5/5 | 1 | |
| de_hub_daily | 3/5 | 4/5 | 1 | |
| ds_cancel_predict | 4/5 | 5/5 | 1 | the base timed out once |
| da_weekend_delay | 0/5 | 1/5 | 1 | base `IN (1, 7)` ×5; adapter `IN (1, 6)` ×3, `IN (6, 7)` ×2 (one with the wrong population) |
| da_delay_deviation | 4/5 | 4/5 | 1 | |
| ds_notam_classify | 5/5 | 4/5 | 1 | the adapter left most test rows without a prediction once |
| ds_taxi_regression | 5/5 | 4/5 | 1 | the adapter timed out once |
| da_redeye_count | 5/5 | 2/5 | 0.17 | the adapter read the hubs' departures as flights from or to them (2), or between them (1) |

The other 12 problems pass 5 of 5 in both states. Statuses over the 115 runs: base 94 ok, 20
wrong, 1 timeout; adapter 103 ok, 11 wrong, 1 timeout. No run failed in the harness or on the
server.

## Reading it

- **Target A's skill is in the adapter and it transfers.** The Target A test showed it on unseen
  tables (10 → 58 of 60). Here it shows on the probe's clinic table and on the aviation warehouse,
  in multi-step agent runs through a harness the training never used.
- **The weekend is the weak point.** In the Target A test the adapter answered every ClickHouse
  weekend item right. In an aggregate on aviation data it mixed the two numberings three times in
  five. The adapter learned "Saturday is 6" more firmly than "Sunday is 7". The doubts it still
  raises about Sunday = 1 (29 of 60 traces in the Target A test) show the same weak spot.
- **Population scope is the one thing that slipped,** on a single problem and without
  significance. It is the reading behind the base's own `da_delay_attribution` finding before the
  prompt was fixed (ADR-004, Target B). Worth watching in the full battery, not a finding yet.
- **k = 5 is small.** A problem has to flip nearly whole to show by itself (0 → 5 gives p = 0.008;
  2 → 5 gives 0.17). The totals point the same way as the per-problem results, but the McNemar
  test over problems, with 4 discordant problems, can't show it (p = 0.63).
