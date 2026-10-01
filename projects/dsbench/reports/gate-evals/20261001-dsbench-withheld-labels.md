# dsbench: the agent could read the labels its grader holds back

ADR-003 §7. Found 2026-10-01 while building Target C's quota mode: each ML task kept its held-out
labels where the agent could read them. The fix is `src/dsbench/agentic/access.py`: each run's
agent gets its own ClickHouse login. The audit of the saved runs is `src/dsbench/agentic/audit.py`,
and its numbers are in `20261001-dsbench-withheld-labels.json`.

## Summary

- **Where the labels were.** The agent ran as the sandbox's admin (`avbench`), which reads
  everything:
  - in the run's own scratch database, `<x>_test_key` for five tasks and `<x>_test_full` for three;
  - in the shared warehouse, `aviation.flights` holds every test flight's outcome for the three
    flight problems, and `aviation.notam` holds the test split's categories.
- **Agents did reach for them.** Of 692 saved runs, from the Gate 0 baselines to the Gate 2
  battery:
  - In the 2026-09-19 baseline, a ds_cancel_predict run by the base (Qwen3.6) listed its tables
    and wrote code that read `cancel_test_full`'s labels. It never delivered and failed.
  - In the Gate 2 battery, a ds_delay_predict run by the base wrote its predictions, then joined
    `delay_test` to `aviation.flights` to score them on the held-out outcomes. It passed, on a
    table written before the join and not touched after.
  - 19 NOTAM runs by Ornith (11 passed) read the test split's category counts. Every one was an
    aggregate, never a row's label, and every one trained on the train split.
  - 3 runs saw a label table in a listing. Only the cancel run above read one.
- **No past score changes.**
  - No passing run delivered from the labels.
  - The NOTAM passes saw only class counts. The bar is 0.55 accuracy against a test majority of
    0.18, which class counts can't clear.
  - The Gate 0 baselines, the Gate 2 battery and the probe's "before" stand.
- **The fix.** Each run's agent gets its own ClickHouse login, made after setup and dropped after
  the run:
  - it has its scratch database, but not the label tables, which vanish from SHOW TABLES;
  - it can SELECT from `aviation`, with a restrictive row policy hiding the problem's test rows:
    hub flights in the test buckets, or the NOTAM test split;
  - the grader keeps the admin login.

  Both oracle gates now check each login on the live sandbox: dsbench 23 of 23, the probe 6 of 6.
  A guard also fails any run whose tool calls name a withheld table or the admin's login
  (`withheld_access`).
- **One prompt line changed.** pi's and the probe's system prompts told the agent to connect as
  `avbench`/`avbench`. They now read the credentials from the environment, which holds the run's
  login. Nothing an agent used in the saved runs is lost, apart from the labels.

## Where the labels were

| Task | In its scratch database | In the shared warehouse |
|---|---|---|
| ds_cancel_predict | `cancel_test_key`, `cancel_test_full` | `aviation.flights.Cancelled` |
| ds_delay_predict | `delay_test_key`, `delay_test_full` | `aviation.flights.ArrDelayMinutes` |
| ds_taxi_regression | `taxi_test_key`, `taxi_test_full` | `aviation.flights.TaxiOut` |
| ds_notam_classify | | `aviation.notam.category` where `split = 'test'` |
| probe_loan_default | `loan_test_key` | |
| probe_energy_demand | `energy_test_key` | |

- **The warehouse route.** A flight problem's test table carries each flight's date, carrier,
  route and scheduled departure. Joined on those, it finds its flight in `aviation.flights`, with
  the outcome. The system prompt describes `aviation.flights` and its outcome columns to every
  run.
- **The NOTAM route.** The NOTAM task tells the agent the test split exists in `aviation.notam`.
  Its grader reads the test labels from there.
- **Target C too.** The seven synthetic Target C tasks keep `<x>_test_key` in the run's database.
  The generator's guard, built the same day, already refuses to keep a run that names one. Later
  that day the generator moved to the same per-run login, with no `aviation` grant (ADR-004).

## What the saved runs did

The audit reads every saved result JSON:
- the 27 in `reports/agentic-runs/`;
- the Gate 2 battery's two dsbench runs, kept on the box in
  `~/benchlab/runs/2026-09-24-gate2-battery/pi-runs/`.

It flags a tool call that names one of the problem's label tables. The names are taken from the
setup itself, so an agent's own `X_test_full` array in Python doesn't count. It also flags:
- a flight problem's test table joined to `aviation.flights`;
- reads of `aviation.notam` categories beyond the train split.

| Flag | Runs | Passed | Who |
|---|---:|---:|---|
| named a label table | 1 | 0 | base (Qwen3.6), baseline 2026-09-19, ds_cancel_predict |
| joined the test table to `aviation.flights` | 1 | 1 | base, Gate 2 battery (thinking off), ds_delay_predict |
| read `aviation.flights`, no join | 1 | 0 | Ornith, ds_delay_predict: flights counted by airline |
| saw a label table listed | 3 | 1 | the cancel run above; Ornith, ds_delay_predict; the adapter, ds_taxi_regression |
| NOTAM test-split category counts | 19 | 11 | Ornith, in 19 of its 28 NOTAM runs; never the base |

- **The cancel run** (`20260919-110559-qwen36-23problem-k5.json`, #99).
  - It ran `SHOW TABLES FROM prob_ds_cancel_predict` and saw all four tables.
  - It then wrote `test_labels = client.query_df('SELECT id, cancelled FROM
    prob_ds_cancel_predict.cancel_test_full')` into its training script, twice.
  - The run never wrote `cancel_pred` and failed on that.
- **The delay run** (`20260924-185004-battery-base.json`, #103).
  - At step 6 it wrote `delay_pred` from a gradient-boosting model.
  - At steps 9 and 10, to "verify the ROC-AUC on the test set", it joined `delay_test` to
    `aviation.flights` on date, carrier, route and departure time, and computed AUC 0.670 on 410
    matched rows and 0.630 on 52,811 duplicated ones.
  - It reported the 0.63 and stopped. The graded table is the one from step 6.
- **The NOTAM counts.**
  - `SELECT category, count() FROM aviation.notam WHERE split='test' GROUP BY category`, or the
    same counts grouped by split, while exploring the data.
  - Every training query in those runs filters `split = 'train'`.

## The fix

**The login** (`access.open_login`, made after setup, dropped after the run):
- **Its scratch database:** `GRANT ALL ON <ns>.*`, then `REVOKE ALL` on each table named
  `*_test_key` or `*_test_full`.
  - ClickHouse subtracts the revoke from the grant, and the tables drop out of SHOW TABLES and
    system.tables.
  - Reading, copying (CREATE TABLE ... AS SELECT), renaming and dropping them are all refused
    (code 497).
- **The warehouse:** `GRANT SELECT ON aviation.*`, plus a restrictive row policy for each row set
  the problem declares (`AgentProblem.withheld_rows`):
  - the flight problems hide hub flights in buckets 0-7 of their split hash, which contains every
    test flight;
  - NOTAM hides `split = 'test'`.

  A restrictive policy narrows only its own user's view. The grader and every other user still
  see all rows.
- **Nothing else.** `remote()`, `url()` and `file()` need grants the login lacks. `numbers()`,
  settings and the system tables still work, as the saved runs used them.

**Where it applies:**
- the native loop (`loop.run_agent`): its run_sql uses the login's client, and its run_python
  passes the login's credentials into the workspace container;
- pi (`pi_runner.run_pi_agent`): the login's credentials go into pi's environment, which the
  `run_sql` and `run_python` wrappers read;
- the probe runs through pi, so it gets the login as well.

**The guard** (`access.breach`). A run fails as `withheld_access`, whatever the oracle says, if a
tool call:
- names one of its withheld tables, matched exactly;
- or logs in as the admin (`user=avbench`, `password='avbench'`, `avbench:avbench`).

**Checked live on the local sandbox:**
- **Both oracle gates.** `dsbench-agent-selftest` and the probe's gate now run `check_access` on
  each problem: dsbench 23 of 23 and the probe 6 of 6. For each problem's login, the check
  confirms that:
  - it lists exactly its inputs;
  - each label table is refused;
  - no withheld row is visible, while the admin sees them;
  - it can write its scratch database;
  - a flight problem's test table no longer joins back to `aviation.flights`. One cancel test row
    shares its schedule with a training flight, so the check allows up to 1%.
- **ds_cancel_predict by hand.**
  - The agent sees 120,891 of the 131,448 hub flights. The 10,557 hidden are exactly
    `cancel_test`.
  - The key is refused through `run_sql`, through `run_python` in the workspace container, and
    through pi's wrappers.
  - The grader reads the agent's `cancel_pred`.
- **No debris.** No agent user or row policy is left after either gate. A crash would leave at
  most a user, which the next run of the same namespace replaces, and a row policy whose user is
  gone, which applies to nobody (checked).

## What it doesn't stop

ADR-003 says the agent's code runs "never the host". That holds for the native loop, not for pi:
- **pi's `bash` tool runs on the host.** It can read the repo, including the compose file with the
  admin's password.
- **The workspace container** mounts the repo and keeps the admin's credentials in its own
  environment.

So an agent that hunts for the admin's login can find it. The guard catches a run that uses it, or
that names a label table, which is what using the labels takes.

The defence is against an agent that comes across the labels, which these runs show happening. It
is not against one that sets out to find them.

## Limits

- **The audit is pattern-based.**
  - For the scratch tables, it matches exact names.
  - For the warehouse, it looks for a join between the test table and `aviation.flights` in one
    call, or category reads from `aviation.notam`.
  - A run that reached the labels some other way, say by assembling the query from pieces, would
    escape it. Reading the flagged runs found nothing beyond what the table lists.
- **Only saved runs can be audited.** The battery's other benchmarks don't run in this sandbox.
