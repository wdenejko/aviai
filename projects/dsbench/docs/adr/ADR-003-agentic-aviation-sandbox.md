# ADR-003: dsbench v2 — an agentic, real-world aviation data-stack sandbox

- **Status:** Accepted. Phase 1 (ClickHouse vertical slice) shipped; **Gate 0 v2 baseline established 2026-09-18** via a public harness (see §6). Phase 2 (live Airflow + MLflow) pending.
- **Date:** 2026-09-15 (baseline addendum 2026-09-18)
- **Deciders:** Wojtek Denejko (box owner)
- **Relates to:** ADR-002 (the single-shot dsbench harness this evolves; its execution-verified / paired / oracle-gated methodology carries over), ADR-001 (the fine-tune study these scores gate), `avtext` (reused data collectors and licensing discipline), memory `reference-fork-thinking-eval` (thinking-model eval protocol), memory `reference-ornith-agentic-behavior` (the baseline finding).
- **Decisions locked with the owner (2026-09-15):** tool-calling agent loop; **all three services live** (ClickHouse + Airflow + MLflow); first increment = ADR + ClickHouse vertical slice; host = the Mac (agent reaches the served model on dashi via the SSH tunnel).
- **Decisions locked with the owner (2026-09-18, Gate 0 baseline):** the v2 baseline is measured through a **public agent harness — `pi`, pinned @ 0.84.4** (not our bespoke loop); **thinking ON** for all agentic tests; **k=5** per-problem pass rates (temp-0 thinking is nondeterministic on the fork); grow the set with harder cases. See §6.

## Context

ADR-002's dsbench asks a model to return one code block or SQL against in-memory fixtures, and grades the returned value. Ornith-1.5 scored 16/21 on it (thinking off), so it discriminates — but it measures *snippet* skill, not the thing the fine-tune study is really about: **operating a real data stack over multiple steps.** The owner wants a benchmark that is *agentic* (multi-step, tool-using, stateful) and *real-world-facing* (a live ClickHouse warehouse, Airflow for orchestration, MLflow for experiment tracking), with every task grounded in **real aviation data**.

This is dsbench v2. It does not replace ADR-002's set — that stays as the fast, cheap regression floor. v2 adds a heavier, higher-signal instrument for the data-engineering / data-analysis / data-science *workflows* the owner actually does.

### What changes from ADR-002

| Dimension | ADR-002 (v1) | ADR-003 (v2) |
|---|---|---|
| Interaction | single-shot: prompt → one code block | **tool-calling agent loop** (ReAct): model calls tools, observes, iterates until done |
| Environment | in-memory pandas/DuckDB fixtures | **live sandbox**: ClickHouse + Airflow + MLflow + a Python workspace, in containers |
| Data | tiny hand-built fixtures | **real aviation datasets** (BTS flights, METAR/TAF, NOTAM), pinned snapshots |
| Answer graded | the returned value | **environment state**: ClickHouse tables, Airflow DAG-run results, MLflow logged runs/artifacts |
| Grading | independent checker recomputes expected | **unchanged in spirit**: an independent oracle recomputes expected state from the raw data |

### What carries over from ADR-002 (non-negotiable)

- **Execution-verified, no LLM judge for the core score.** We assert on real state produced by running the agent's work.
- **Oracle gate** (v1's `dsbench-selftest`): a reference solution must reach the graded state before any model is scored. If the reference fails, the bug is in the problem.
- **Independent expected computation.** The checker recomputes the answer from the raw source a *different* way — never by trusting the agent's output.
- **Paired, noise-aware comparison** (`dsbench-compare`) and a **fair, discriminating** checker (negative controls) remain the standard.
- **Thinking-model protocol** (memory `reference-fork-thinking-eval`): v1 runs **thinking off** at temp 0 for a reproducible, artifact-free loop. **The v2 agentic baseline flips this to thinking ON** (owner's call, 2026-09-18) — agentic coding is a thinking-on use case, and the fork serves it cleanly via `chat_template_kwargs enable_thinking=true`. temp stays 0; the run records the protocol. See §6.

## Decision

Build **`projects/dsbench/sandbox/`**: a Docker-Compose stack on the Mac plus an agent harness that drives a tool-calling loop against it, with aviation problems graded on the resulting stack state.

### 1. The sandbox (docker-compose on the Mac)

Four services on one Docker network (`avnet`):

- **`clickhouse`** — the warehouse. Holds the aviation datasets as tables; the agent's `run_sql` runs here. Memory-capped (`max_server_memory_usage`) so it coexists with the model tunnel and Airflow/MLflow.
- **`workspace`** — a Python container with the DS/DE/DA stack (pandas, polars, numpy, scikit-learn, scipy, pyarrow, clickhouse-connect, mlflow client, apache-airflow client, matplotlib). The agent's `run_python` executes **here**, never on the host. A repo dir is mounted for the agent's files and for Airflow `dags/`.
- **`airflow`** — Airflow in `standalone` mode (scheduler + webserver + SQLite/Postgres metadata), `dags/` bind-mounted from the workspace so a DAG the agent writes is picked up and can be triggered and its run asserted on.
- **`mlflow`** — an MLflow tracking server (backend store + local artifact dir on a volume). The agent logs runs; the checker queries the tracking API for logged params/metrics/artifacts.

Canonical spec is a portable `docker-compose.yml` (runs anywhere with Docker/Compose); on the Mac it runs as-is. (dashi has podman but no compose provider; if we ever colocate, translate via `podman play kube` — out of scope here.)

### 2. The agent loop (tool-calling ReAct)

`dsbench-agent` (new) drives, per problem:

1. Sends the problem prompt + tool schemas to the served model over `/v1/chat/completions` (the same OpenAI seam as v1's `ChatClient`), pointed at the model on dashi via the tunnel (`http://localhost:18080`), **thinking off, temp 0**.
2. Parses tool calls, executes them, returns observations, loops — up to a **step budget** (`--max-steps`, default ~20) and a wall-clock cap.
3. Tools (v2 minimum): `run_sql(query)` → ClickHouse rows; `run_python(code)` → exec in `workspace`, stdout/stderr/return; `write_file(path, content)` / `read_file(path)` in the mounted repo; `trigger_dag(dag_id, conf)` + `get_dag_run(run_id)` for Airflow; `log_note`/finish. MLflow is used by the agent's own Python (its tracking URI is preset), not a bespoke tool.
4. Records the full trajectory (messages, tool calls, observations) for debugging and offline re-grading.

**Tool-call format:** the served Qwen3.x template emits tool calls in the Qwen XML form (ADR-001 flags the `qwen3_xml` vs `qwen3_coder` parser and the `arguments|items` template bug). The harness must parse the format the fork actually serves; a parse failure is a recorded, distinct status (like v1's `truncated`), not a silent wrong.

**Streaming, not block, for robustness (resolved 2026-09-15).** Native OpenAI tool-calling works on the fork, but the **non-streaming** endpoint runs a strict `json::parse` over the model's tool-call `arguments` and **500s the entire request** when the model emits one malformed/truncated call deep in a long trajectory — which killed `ds_notam_classify` (~step 15). A controlled probe disproved the tempting "double-quotes break it" theory: the fork escapes double-quoted code and parses multi-KB generated args fine on both paths. The fix is entirely harness-side and preserves native tool-calling: **stream the completion and reassemble tool-call arguments from the deltas** (the stream path has no such gate), treat a still-unparseable reassembled call as a **recoverable tool error** (fed back so the model resends), and write a sanitized valid-JSON copy of every call into the resent history so the next turn's prompt render can't 500 either. A model that occasionally fumbles a tool call is the fine-tune signal we want to *measure*, not a crash that voids the run. `agentic/loop.py`: `_reassemble_stream`, `call_model(stream=True)`.

### 3. Grading = independent oracle over final state

Each problem defines `setup()` (idempotent per-problem namespace — a dedicated ClickHouse database / MLflow experiment / DAG id, so problems don't collide and re-runs are clean), `check()` (reads the *raw* source, computes expected **independently**, and asserts against the stack state the agent produced), and a `reference` (a script/solution that reaches the graded state — powers the oracle gate). Statuses extend v1's: `ok | wrong | error | timeout | no_answer | parse_error | setup_error`.

### 4. Datasets (pinned snapshots, aligned window + airports)

To enable cross-source problems (flights × weather × NOTAMs), all three align on **one month** and a fixed **airport set** (a handful of busy US hubs, e.g. ORD/ATL/DFW/DEN/LAX). Loaded once into ClickHouse; version + checksum pinned in a manifest.

- **BTS "Reporting Carrier On-Time Performance"** — direct GET `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_${Y}_${M}.zip` (Appendix A). US-gov **public domain** → publishable. One month ≈ 500–600k rows.
- **METAR/TAF** — reuse `avtext/src/avtext/ingest/{iem,awc,iem_taf,awc_taf}.py` (IEM ASOS archive + AWC). Underlying obs US public-domain; commit manifests not bulk.
- **NOTAM** — candidate publishable path noted here was Zenodo Pik 2023 (`11420433`), CC-BY 4.0; avtext's `opennotam`/`knots` sets are **quarantined (eval-only, never redistributed)** and may be used *locally* only. **Implemented (Phase 1): DEEL-AI/NOTAM (HuggingFace), MIT** — an 8,478-row, 13-class text-classification corpus. It has no operational fields (ICAO/Q-code/effective window) and is static ~2024, so it does **not** join to flights/weather; it powers a standalone NOTAM text-classification task instead of a cross-source one. Only the manifest + loader are committed (records stay local).

### 5. Layout

```
projects/dsbench/sandbox/
  docker-compose.yml            # clickhouse (P1) + workspace; airflow + mlflow (P2)
  README.md                     # bring-up, teardown, safety
  clickhouse/ init/*.sql        # schemas: flights, metar, taf, notam, airports
  ingest/                       # bts.py (new); metar/taf/notam adapters reusing avtext
  manifests/                    # pinned dataset versions + checksums
projects/dsbench/src/dsbench/agentic/    # as shipped in Phase 1
  loop.py tools.py ch.py        # ReAct loop (+ grading) · tool executors · ClickHouse client
  schema.py                     # AgentProblem: setup/check/reference/budget; GradeContext; AgentResult
  loader.py                     # discover problems/{de,da,ds}/*
  problems/{de,da,ds}/          # aviation agentic problems
  runner.py                     # dsbench-agent-run
  selftest.py                   # dsbench-agent-selftest — oracle gate (references reach graded state)
```

### 6. Gate 0 v2 baseline: a public harness (`pi`), thinking on, k=5 (decided 2026-09-18)

The Phase-1 baseline (§action-item 6) was measured through **our own** ReAct loop, and its headline miss
(`da_hub_delay`) turned out to be an artifact of *that loop's* `finish(answer=)` contract, not the model.
For a Gate-0 number that is **accurate and portable** — attributable to a mainstream agent and reproducible
by others — the owner directed three changes:

- **Measure through a public agent harness — [`pi`](https://github.com/badlogic/pi-mono) (`earendil-works/pi`, MIT), pinned @ `0.84.4`.** Native tool-calls (`read`/`bash`/`write`), headless `--print --mode json --no-session`, and a `--thinking` flag. The baseline number is defined as the triple **"model + pi 0.84.4 + thinking level"** — pin all three when quoting a score. (Our `agentic/loop.py` stays as a debugging/inspection instrument, not the scoring path.) `pi` drives the *same* sandbox through thin `sandbox/pi/bin/{run_sql,run_python}` wrappers, and `dsbench-pi-run` (`agentic/pi_runner.py`) orchestrates per-problem runs and grades the end state with the **same independent oracle** — capability misses (`wrong`) are split from tool-reliability failures (`model_error`). Provider config: `~/.pi/agent/models.json` → `dashi-ornith` (baseUrl `http://localhost:18080/v1`, thinkingFormat `qwen-chat-template`, temp 0).
- **Thinking ON.** Agentic coding is a thinking-on use case; sweeps run `--thinking high`. (Per-request `reasoning_effort` is ignored by the fork — the server is launched `--reasoning-effort medium` — but the chat-template thinking toggle works.)
- **k=5 per-problem pass rates.** temp-0 **with thinking is not deterministic** on this fork (MoE routing + parallel KV slots), so a single run misreports borderline problems. The baseline reports **passes/k and majority-pass** over k=5. This is the single biggest accuracy improvement over the Phase-1 1/2 number.

**Set grown to 10 (harder cases).** Added `da_delay_deviation` (trailing-7-day window), `de_carrier_ontime` (group + `row_number()` rank), `de_taf_latest` (`argMax` dedup), and two deliberate **ClickHouse-dialect traps** (`da_weekend_delay`, and the existing `da_cancel_dow`) that hinge on the `dayOfWeek`/`toDayOfWeek` ISO convention (1=Monday). All 10 pass the oracle gate (`dsbench-agent-selftest` 10/10). Prompts were made harness-agnostic (no "call finish" wording).

**Set expanded to 20 (2026-09-18) — headroom for the fine-tune.** 8/10 by majority left too little room to *measure* an improvement, so 10 more hard problems were added across five capability axes, each with a pandas-vs-ClickHouse cross-validated oracle (gate now **20/20**): **time/timezone** — `da_worst_dep_hour` (parse a String `hhmm`), `da_utc_peak_hour` (per-hub local→UTC, sign of the offset), `da_redeye_count` (midnight wraparound); **dialect/NULL** — `da_all_flights_avg_delay` (`avg` skips NULLs → wrong denominator); **statistical rigor** — `da_weighted_ontime` (pooled vs average-of-averages / Simpson), `da_delay_attribution` (5-cause share, population rule); **window / gaps-and-islands** — `da_delay_streak` (longest consecutive-day run); **grain + rank** — `da_cancel_weather_share` (rate within a group), `de_route_leaderboard` and `de_recovery_leaderboard` (state tables, conditional denominators). Candidates deliberately *dropped* after probing the data (rigor over volume): a `uniq()`-approximation trap (ClickHouse `uniq` is exact at ~1.2k distinct here), an exact-percentile trap (`quantileExact` vs pandas interpolation is oracle-fragile), and a METAR as-of join (obs cadence is ~5 min, so any as-of/tz error still matches an obs → non-discriminating).

**Set expanded to 23 (2026-09-19) — the DS/ML axis.** Rounded out DS (was one text-classification problem) with three more leakage-safe ML tasks so it spans the canonical types: `ds_delay_predict` (binary late-arrival, **ROC-AUC** — the target is zero-inflated so MAE/accuracy are traps, only ranking is honest), `ds_cancel_predict` (**imbalanced** ~2% binary, ROC-AUC), `ds_taxi_regression` (**regression**, MAE vs the predict-the-median baseline — taxi-out is not zero-inflated, so MAE is fair). Each exposes only pre-departure features (leakage-safe by construction), uses a deterministic `cityHash64` split with a once-materialised test id, and grades held-out truth the agent never sees. Gate **23/23**. The authoritative baseline below is now the **23-problem k=5** run (superseding the earlier 10-problem 37/50).

**Reframing — the real fine-tune target is capability, not plumbing.** Under `pi`: `da_hub_delay` passes, **zero** `model_error`s, and the write-heavy `ds_notam_classify` (94% CV) is solid. The Phase-1 "structured-tool-output is the bottleneck" conclusion was a native-`finish` artifact. The residual, *reproducible* weakness is **SQL-dialect capability** — two independent problems isolate the ClickHouse `dayOfWeek` convention as the sharpest miss. This is the movable Gate-2 target for ADR-001.

**Authoritative baseline — Ornith-1.5-35B-A3B (Q8_0), pi 0.84.4, thinking high, temp 0, k=5** (`reports/agentic-runs/20260919-005921-ornith-23problem-k5.json`):

**Overall: 75/115 passing runs · 16/23 problems pass by majority.**

| Problem | Cat | passes/5 | Majority | Note |
|---|---|---|---|---|
| `da_hub_delay` | da | 5/5 | ✅ | |
| `da_worst_dep_hour` | da | 5/5 | ✅ | String `hhmm` parse |
| `da_all_flights_avg_delay` | da | 5/5 | ✅ | NULL/denominator |
| `da_weighted_ontime` | da | 5/5 | ✅ | pooled, not avg-of-avgs |
| `da_cancel_weather_share` | da | 5/5 | ✅ | rate within group |
| `da_redeye_count` | da | 4/5 | ✅ | 1 undercount (5333 vs 6538) |
| `da_delay_deviation` | da | 4/5 | ✅ | 1 wrong (DEN 06-08) |
| `da_delay_streak` | da | 3/5 | ✅ | 2 empty answers |
| `da_ts_delay` | da | 3/5 | ✅ | 2 format/variance |
| `da_cancel_dow` | da | **0/5** | ❌ | **`dayOfWeek` convention** (picks Wed, not Thu) |
| `da_delay_attribution` | da | **0/5** | ❌ | **systematically 42.8% vs 44.1%** (wrong denominator/population) |
| `da_utc_peak_hour` | da | **1/5** | ❌ | **timezone conversion** (got 5, not 13) |
| `da_weekend_delay` | da | **1/5** | ❌ | **`dayOfWeek` trap** (`IN (1,7)` = Mon+Sun) |
| `de_hub_daily` | de | 5/5 | ✅ | cross-source fact table |
| `de_taf_summary` | de | 5/5 | ✅ | |
| `de_recovery_leaderboard` | de | 5/5 | ✅ | conditional denominator + rank |
| `de_carrier_ontime` | de | 4/5 | ✅ | 1 wrong grain (86 rows) |
| `de_route_leaderboard` | de | 4/5 | ✅ | 1 table-write miss |
| `de_taf_latest` | de | 3/5 | ✅ | 2 wrong (returned 1618 rows) |
| `ds_notam_classify` | ds | 5/5 | ✅ | 13-class text ML |
| `ds_cancel_predict` | ds | **1/5** | ❌ | **ML workflow: didn't deliver the pred table** |
| `ds_delay_predict` | ds | **1/5** | ❌ | **ML workflow: didn't deliver the pred table** |
| `ds_taxi_regression` | ds | **1/5** | ❌ | **ML workflow: over-tuned, never wrote the table** |

The 7 majority-failures cluster into three **movable fine-tune targets**, not noise:
1. **SQL-dialect conventions** — `da_cancel_dow` + `da_weekend_delay` (ClickHouse `dayOfWeek` ISO 1=Monday) and `da_utc_peak_hour` (local→UTC offset sign). The sharpest, most reproducible gap.
2. **A systematic reasoning error** — `da_delay_attribution` returns **42.8% every single run** vs the correct 44.1%; a consistent wrong denominator/population reading (not variance), which is the cleanest possible signal.
3. **Agentic ML-workflow discipline** — `ds_delay_predict` / `ds_cancel_predict` / `ds_taxi_regression` all land at 1/5, but the trajectories show the model is *competent at the ML* (a failed `ds_taxi_regression` run tuned GBR to MAE 6.75 < 7.64 baseline in log-space) — it just **fails to deliver the output table within the run**: it over-explores/tunes or ends its turn before writing. `ds_notam_classify` at 5/5 proves the write path works when the model commits the artifact. This is the direct descendant of the old "structured-output reliability" theme, now on real ML tasks.

The remaining 1–2/5 misses on otherwise-passing problems (an undercount, an empty answer, a wrong grain, `de_taf_latest` returning all rows) are run-to-run variance — a secondary reliability signal, and exactly why k>1 is mandatory.

**Harness gaps fixed en route** (all setup, not model misses — each surfaced as a false negative until pinned): `run_python` must inject the full `CLICKHOUSE_*` env (host `clickhouse`, avbench/avbench) or the agent connects to `localhost` inside the container and fails; `SYSTEM_PI` must document the write API (`client.command('CREATE TABLE … ENGINE=MergeTree ORDER BY …'); client.insert_df('t', df)`) or the model invents `bulk_insert`/`data_insert` and 0/5s a solvable problem; and prompt ambiguities (cancellation handling; "n_periods" all-rows vs Forecast-only) must be pinned with an exact, discrete/rounded expected answer.

## Options considered (the owner's choices, with the trade-off recorded)

- **Agent loop vs single-shot** → *agent loop.* Real-world-facing and what the fine-tune targets; cost is a harder harness, tool-format fragility, and looser determinism (mitigated by step budgets, temp-0/no-think, final-state grading).
- **All three live vs artifact-validated** → *all three live.* Maximum realism and the truest signal; cost is the heaviest, slowest, most nondeterministic sandbox and real ops surface (Airflow scheduler, MLflow server). Mitigated by per-problem namespaces, idempotent setup/teardown, and grading state not trajectory. Phase 1 stands up ClickHouse only; live Airflow + MLflow land in Phase 2 so the harness and grading model are proven cheaply first.
- **Mac vs dashi host** → *Mac.* Keeps the GPU box lean (it just serves); Docker is up on the Mac. Cost is cross-network to the model (fine over the tunnel) and that the sandbox is not colocated with the data-heavy box. The datasets live in-repo on the Mac already.
- **Reuse dsbench vs new project** → evolve dsbench (`sandbox/` + `agentic/`); may graduate to its own project if it outgrows the v1 harness.

## Consequences

- **Easier:** a benchmark that measures the owner's actual workflows; cross-source aviation problems; the same instrument scores no-training alternatives (better prompts, retrieval); v1 stays as a fast regression floor.
- **Harder / new surface:** a container stack to run and keep deterministic; an agent loop and a tool-call parser tied to the served template; live-service state to reset between problems; the agent executes arbitrary Python + SQL — **contained to the `workspace`/`clickhouse` containers, never the host**, network-restricted, resource-limited (v1's "own model, own box" trust model, now enforced by the container boundary — do not point this at an untrusted endpoint).
- **Determinism:** live services + an agent are less reproducible than v1. Controlled by pinned data snapshots, temp-0 / thinking-off, per-problem namespaces, idempotent setup, step/time budgets, and final-state (not trajectory) grading. Conclusions still come from paired flips + McNemar, not a single percentage.
- **Licensing:** publishable stack = BTS (public domain) + IEM/AWC METAR/TAF + Zenodo NOTAM (CC-BY, attributed). Quarantined avtext NOTAM sets never leave local eval. Only manifests/loaders are committed — bulk records never enter git.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Tool-call format mismatch with the fork's template | Med | Agent can't act | Parse the served Qwen XML form; `parse_error` status; test against Ornith early |
| Live-service nondeterminism → flaky grades | Med | Spurious flips | Per-problem namespaces, idempotent setup/teardown, final-state grading, fixed snapshots |
| Sandbox memory pressure (Mac) with 4 services | Med | Slow / OOM | Cap ClickHouse memory; Airflow standalone; MLflow light; bring services up per phase |
| BTS PREZIP schema/URL drift or TLS quirks | Low | Ingest breaks | Pin the month + checksum a manifest; relaxed TLS; snapshot committed-by-manifest |
| NOTAM operational fitness (avtext sets are extraction-labeled) | Med | Weak NOTAM problems | Evaluate Zenodo CC-BY set for operational fields; align to the chosen month/airports |
| Arbitrary code execution escaping the sandbox | Low | Host risk | Exec only inside containers; no host mounts beyond the repo workdir; network-restricted; own-model-only |
| Agent overfits the exact tool wrapper | Low | Unfair scores | Keep tool schemas stable + documented; grade outcomes, not tool syntax |

## Action items — phased

**Phase 1 — ClickHouse vertical slice:**
1. [x] `sandbox/docker-compose.yml` with `clickhouse` + `workspace`; up + verified (fixed `listen_host`).
2. [x] `ingest/bts.py` — June 2026 via PREZIP → `aviation.flights` (607,577 rows); manifest + sha256.
3. [x] All datasets loaded: METAR (`ingest/metar.py`, IEM, 45,796 obs) + `aviation.airports`; **TAF** (`ingest/taf.py`, IEM, 9,060 decoded forecast periods / 1,618 bulletins, June 2026); **NOTAM** (`ingest/notam.py`, DEEL-AI, MIT, 8,478 records, 13-class — a static ~2024 corpus, not date-aligned, standalone).
4. [x] `agentic/` harness: ReAct loop + `run_sql`/`run_python`/`finish`; **native OpenAI tool-calling** (verified working on the fork — the ADR-001 template worry did not materialise); thinking-off; trajectory logging; `model_error`/`truncated` statuses.
5. [x] 6 problems across de/da/ds using all five tables — answer- and table-graded, incl. a NOTAM-ML task with a `setup()`-seeded unlabeled test set graded on real held-out accuracy; independent oracles + negative controls; `dsbench-agent-selftest` green 6/6.
6. [x] Baseline Ornith-1.5 (native loop, thinking off): **1/2** — but its `da_hub_delay` "fail" was a `finish(answer=)` artifact, not the model. **Superseded by the Gate-0 baseline in §6** (public harness, thinking on, k=5): **37/50 runs, 8/10 by majority.**

**Phase 1 + Gate 0 complete** (data foundation + agent harness + **10** problems + an authoritative public-harness baseline; see §6). Next: **Phase 2 — live Airflow + MLflow** (problems that build/trigger a DAG and log an experiment, graded on live service state); keep growing the set and re-baseline as it grows.

**Phase 2 — live Airflow + MLflow:** add both services; `trigger_dag`/`get_dag_run` tools; MLflow tracking URI preset in `workspace`; problems that build a DAG (assert on the run) and log an experiment (assert on logged metrics/artifacts).

**Phase 3 — grow + wire to ADR-001 gates:** more problems per (category), cross-source tasks, and use v2 as a Gate-2 before/after instrument alongside v1.

---

## Appendix A — BTS collection (researched 2026-09-15)

- **Endpoint (keyless GET):** `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_${YEAR}_${MONTH}.zip` — `MONTH` is 1–12, **not** zero-padded. Table `gnoyr_VQ=FGK` on the DL_SelectFields form; the PREZIP file is the full field set (bypasses the form POST).
- **Contents:** one CSV named `On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_${YEAR}_${MONTH}.csv` + a `readme.html`. ~110 columns; ~500–600k rows/month domestic.
- **Practicalities:** transtats historically needs relaxed TLS (`--no-check-certificate` / `verify=False`) and a descriptive User-Agent + polite rate. Pick a recent **complete** month and pin it in the manifest.
- **License:** US-gov public domain → safe to redistribute (unlike the quarantined NOTAM sets).
- **Key fields for problems:** `FlightDate, Reporting_Airline, Origin, Dest, CRSDepTime, DepDelay, DepDelayMinutes, TaxiOut, WheelsOff, ArrDelay, ArrDelayMinutes, Cancelled, CancellationCode, Diverted, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay` — enough for delay/cancellation analysis and, joined to METAR/TAF/NOTAM on airport+time, cross-source workflows.

## Appendix B — reused avtext assets

- Collectors: `avtext/src/avtext/ingest/{iem,iem_taf,awc,awc_taf}.py` (METAR/TAF), `{opennotam,deel_notam}.py`, `select_stations.py`.
- Licensing discipline: every source pre-cleared before use; `data/third_party/` quarantine = eval-only, never committed/trained/redistributed. v2 commits only manifests/loaders, never bulk records.
- Station registry: OurAirports CSV (public domain) for airport metadata / ICAO↔IATA mapping used to align datasets.
