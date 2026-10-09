# The held-out probe and dsbench: Revision 2.1 against the base, through pi

ADR-004, item 12. Run 2026-10-08/09 on dashi, in one window of 3 hours 40 minutes. The base and
Revision 2.1's final adapter go through the owner's pi harness, k = 5 runs a problem, with
thinking on. The setup is Revision 2's (`battery_window.sh hold:probe`, `rev2_probe_mac.sh`,
`agentic/compare_runs.py`).

Revision 2's runs (2026-10-06/07) give two readings across windows: Revision 2 against Revision
2.1, and the base against itself (A/A). The runs are in `reports/agentic-runs/`. The comparisons
are in `20261009-rev2-1-probe-dsbench.json`.

## Summary

- **The probe: Revision 2.1 passes 29 of 30 runs,** against this window's base's 23. Revision 2
  passed 28.
  - The weekday task goes from 1 of 5 to 5 of 5 (Fisher p = 0.048). The adapter writes
    `toDayOfWeek(visit_date) = 2` every time. The base wrote MySQL's Tuesday, `= 3`, four times.
  - The timezone task goes from 2 to 4 of 5. The adapter's one miss, like the base's three,
    answers hour 4 for 3: it converts in the wrong direction (recomputed from the task's data:
    local minus the offset peaks at 4).
- **dsbench: no gain over this window's base,** 95 → 96 of 115 runs, and 18 → 19 of 23
  problems by majority. Revision 2 gained 94 → 103 in its own window.
- **`da_weekend_delay`, the problem the round was for, moves a little:** 0 → 2 of 5 (Revision 2:
  1 of 5).
  - Revision 2's half-convention, `IN (1, 6)` in 3 of its 5 runs, is gone.
  - Revision 2.1 writes ISO's `IN (6, 7)` in 3 runs: 2 pass, and 1 takes the wrong population.
  - Its other 2 runs write MySQL's `IN (1, 7)`.
- **`da_cancel_dow` falls back to the base's level:** 2 of 5, as for this window's base (Revision 2:
  5 of 5).
  - The 3 misses answer Wednesday: they read `toDayOfWeek`'s 4 with Sunday = 1.
  - One reading, from one problem at k = 5: Revision 2.1 holds the convention more firmly from a
    weekday's name to its number (the probe, the Target A test) than from a number back to a name.
- **Whether Revision 2.1 is worse than Revision 2 on dsbench can't be told from these runs.**
  - Across windows, 103 → 96 runs (Fisher p = 0.25). By majority, 3 problems are lost and 1
    gained (McNemar p = 0.63).
  - The base itself moves this much between windows. The A/A has `da_redeye_count` 5 → 2,
    `da_ts_delay` 5 → 3 and `da_cancel_dow` 0 → 2, with the same totals (94 and 95).
- **The A/A settles one of Revision 2's findings.** Revision 2's `da_redeye_count` went 5 → 2
  against its base. This window's base gets 2 of 5 too. All 3 of its misses read the hubs'
  departures as flights from or to them, the reading that took 2 of Revision 2's 3 misses. That
  drop was the base's spread, not the adapter.
- **Decision 10 (ADR-004) is open:** whether Revision 2.1 replaces Revision 2. The proposal is to
  run the two adapters head to head in one window first, with more runs on the five problems that
  moved.

## What was run

- **The window** (`~/benchlab/runs/2026-10-08-qwen36-rev2-1-probe/`): `battery_window.sh` with one
  step, `hold:probe`, serving the base with Revision 2.1's final adapter loaded
  (`2026-10-08-qwen36-rev2-1-gate-final/lora.gguf`).
  - **Start:** `probe_chain.sh` armed it behind the tests window's third launch (`TESTS_LOG` set
    to that window's log). It started when the tests window ended.
  - **Box clock:** 21:59 to 01:40 (CEST 23:58 to 03:39).
  - **Box state:** production was off, and OCR was restored at the end.
- **Parity, first:** scale 0 gave the bare base's greedy replies on all 8 prompts, and again after
  scale 1. Scale 1 changed 2 of them, as in the tests window.
- **The Mac:** `rev2_probe_mac.sh` with `LABEL=rev2-1`, from a clean clone of the merged code. It
  was restarted at 22:12 CEST after the owner's use of the box, and picked up the hold at 23:59.
  - **Steps:** probe base 9 minutes, probe adapter 11, dsbench base 91, dsbench adapter 108.
  - **Scale:** it set the server's scale and read it back before each step. At the end it put it
    back to 0 and released the hold.
- **pi's settings,** the owner's and Revision 2's: pi 0.84.4, thinking on, temperature 0, 12,288
  tokens a reply. The script checks the reply cap before it starts.
- **Grading:** each problem's own oracle, on the Mac's ClickHouse sandbox, with the agent on its
  own login. After the window, both oracle gates passed without a model: the probe 6 of 6, dsbench
  23 of 23.

## The probe

| Problem | Skill | Base 1 | Revision 2 | Base 2 | Revision 2.1 | Fisher p, base 2 vs 2.1 |
|---|---|---:|---:|---:|---:|---:|
| probe_weekday_visits | dialect weekday | 0/5 | 4/5 | 1/5 | **5/5** | 0.048 |
| probe_utc_peak_trips | timezone direction | 5/5 | 4/5 | 2/5 | 4/5 | 0.52 |
| probe_incident_share | population share | 5/5 | 5/5 | 5/5 | 5/5 | 1 |
| probe_pooled_rating | pooled rating | 5/5 | 5/5 | 5/5 | 5/5 | 1 |
| probe_loan_default | agentic ML (ROC-AUC) | 5/5 | 5/5 | 5/5 | 5/5 | 1 |
| probe_energy_demand | agentic ML (MAE) | 5/5 | 5/5 | 5/5 | 5/5 | 1 |
| **Runs** | | 25/30 | 28/30 | 23/30 | **29/30** | |

- **The base's timezone task moves between windows:** 5 of 5 on 2026-10-06, 2 of 5 now, and 3 of
  5 on 2026-09-20. Every one of its misses gets hour 4 for 3, the wrong direction.
- **The weekday task is the probe's test of Target A.** It is the skill on a domain neither
  dsbench nor the training holds, and it is 5 of 5 here.

## dsbench

The problems that any of the four runs fails:

| Problem | Base 1 | Revision 2 | Base 2 | Revision 2.1 | What the misses are |
|---|---:|---:|---:|---:|---|
| da_weekend_delay | 0/5 | 1/5 | 0/5 | 2/5 | Revision 2.1: `IN (1, 7)` twice; `IN (6, 7)` with the wrong population once |
| da_cancel_dow | 0/5 | 5/5 | 2/5 | 2/5 | every miss names Wednesday: 4 read with Sunday = 1 |
| da_utc_peak_hour | 2/5 | 4/5 | 2/5 | 2/5 | every miss gets hour 5 for 13: the direction trap |
| de_carrier_ontime | 2/5 | 5/5 | 2/5 | 2/5 | ranks not contiguous, or on-time rates off, in both states |
| da_redeye_count | 5/5 | 2/5 | 2/5 | 3/5 | the hubs' departures read as flights from or to them (10,033), or between them (590) |
| da_ts_delay | 5/5 | 5/5 | 3/5 | 4/5 | base 2: 22.9 for 32.5 on the ts flights, twice; Revision 2.1: a timeout |
| da_all_flights_avg_delay | 5/5 | 5/5 | 5/5 | 3/5 | Revision 2.1 answers 22.2 for 22.14 twice: rounded past the tolerance |
| da_delay_deviation | 4/5 | 4/5 | 5/5 | 5/5 | |
| de_hub_daily | 3/5 | 4/5 | 4/5 | 4/5 | |
| de_route_leaderboard | 4/5 | 5/5 | 5/5 | 5/5 | |
| ds_cancel_predict | 4/5 | 5/5 | 5/5 | 5/5 | |
| ds_notam_classify | 5/5 | 4/5 | 5/5 | 5/5 | |
| ds_taxi_regression | 5/5 | 4/5 | 5/5 | 4/5 | a timeout |
| **Runs** | 94/115 | 103/115 | 95/115 | 96/115 | |
| **Problems by majority** | 19/23 | 21/23 | 18/23 | 19/23 | |

The other 10 problems pass 5 of 5 in all four runs. No run failed in the harness or on the server.

### The two weekday problems, run by run

- **`da_weekend_delay`** asks for the average delay on weekends and on weekdays.
  - **Base 2:** `toDayOfWeek(...) IN (1, 7)` or `dayOfWeek(...) IN (1, 7)` in all 5 runs, MySQL's
    weekend.
  - **Revision 2:** `IN (1, 6)` 3 times, ISO's Saturday next to the old Sunday. `IN (6, 7)` twice:
    one pass and one with the wrong population.
  - **Revision 2.1:** `IN (6, 7)` 3 times, with 2 passes. The third run's weekday average is right
    but its weekend average is off (23.6 for 24.3), the wrong population. `dayOfWeek(...) IN
    (1, 7)` twice.

  The round's weekend rows removed the half-convention. In two runs out of five the old one came
  back whole. The convention reaches the right days in 3 of 5 runs, against Revision 2's 2.
- **`da_cancel_dow`** asks for the weekday with the highest cancellation rate. Every run computes
  rates by `toDayOfWeek(FlightDate)` and then names the top number.
  - **Thursday** is 4 in ISO, ClickHouse's numbering.
  - **Wednesday** is what 4 means under Sunday = 1.
  - **The tally:** Revision 2 named Thursday 5 times. Revision 2.1 named it twice and Wednesday 3
    times, as did this window's base. Base 1 named Wednesday every time.

## Reading it

- **The probe confirms the transfer, on a domain the training never saw.** Revision 2.1's weekday
  task is 5 of 5, as Revision 2's was 4 of 5. Its timezone task holds.
- **On dsbench, the round's effect is small and mixed.**
  - The weekend half-convention is gone: 2 of 5 pass, against Revision 2's 1.
  - The weekday name falls back to the base's level in 3 runs.
  - Overall there is no gain over this window's base. The other changes are presentation and
    reading slips (a rounding, rank numbering, populations) that the base makes as well.
- **These runs can't rank Revision 2 against Revision 2.1 on dsbench.**
  - Each adapter was run in its own window, and the base's per-problem results move by up to 3
    runs in 5 between windows.
  - The difference sits on a few problems at k = 5. `da_cancel_dow` and `de_carrier_ontime` are
    each 5 → 2 of 5 (p = 0.17).
  - What would decide it: both adapters on one server, in one window, with more runs on the
    problems that moved. That means `da_cancel_dow`, `da_weekend_delay`, `de_carrier_ontime`,
    `da_utc_peak_hour` and `da_all_flights_avg_delay`.
- **On every test so far, Revision 2.1 does at least as well as Revision 2,** except dsbench, where
  it can't be told apart. It is better on DuckDB's weekday function, and as good on the gate,
  ClickHouse and the probe. dsbench uses no DuckDB.

## Note added 2026-10-09: the server's prompt cache crossed states

Found the same day, from this window's server log (`dsbench.battery.cache_audit`). It is the
same as in Revision 2's window.
- **Two probe problems were affected.** In the adapter's step, all five runs of
  `probe_incident_share` and of `probe_energy_demand` started from 1,251 and 1,241 of their
  prompt tokens as the base had computed them, in the base's step.
  - llama-server loaded those prompts from its RAM copy, which matches by tokens alone.
  - pi's requests name no adapter, so the server never dropped them.
  - Both problems pass 5 of 5 in every run here, so no count changes.
- **Nothing else was affected.** No dsbench run reused anything across steps, and the weekday and
  timezone tasks computed every prompt at their own state.
- **The parity check missed it.** In its scale-1 step, 3 of its 8 prompts started from their own
  scale-0 KV. It compared only the two scale-0 steps, so it passed; it now also counts cached
  tokens.
- **The fix** (`patches/README.md`, "The prompt cache across states"): the server keeps no RAM
  copy when it serves an adapter, and `set-scale` erases every slot. The head-to-head of decision
  10 runs with both.
