# sftgen — targeted SFT-data generators

The sandbox that **measures** the model (ADR-003) also **manufactures** its fine-tune data here,
under the same discipline: every emitted row carries an execution proof. Design and rationale live
in [ADR-004](../../../docs/adr/ADR-004-targeted-sft-data.md); this file is the quick-start + status.

## Why

Gate 0 (ADR-001) found the base (Qwen3.6-35B-A3B) and Ornith share three **family-wide** gaps. Each
generator here targets one, teaching the **transferable** skill on a distribution that is *not* the
aviation benchmark, so a dsbench before/after measures learning, not memorisation.

| Target | Generator | Skill | Status |
|---|---|---|---|
| **A** | `dialect_conventions.py` | SQL-dialect date/time conventions (weekday, weekend, tz, month) | **done** — 6 domains × 4 families, execution-verified, teacher-free |
| **B** | `denominator_reasoning.py` | correct conditional-population / ratio denominator | **built, then DROPPED** — no measured gap (probe 5/5 + `da_delay_attribution` was a prompt-scope artifact). Generator kept as a tool; not in the mixture. See ADR-004 |
| **C** | `ml_delivery_trajectories.py` | agentic ML-workflow delivery discipline | **done** — runs the teacher as a sandbox agent on synthetic ML tasks; keeps only oracle-passing trajectories |
| — | `decontaminate.py` | 13-gram + schema-identifier + numeric-answer gate vs dsbench | **done** — all 3 rules verified |

## Run Target A

```bash
# generate (uses every reachable dialect engine: DuckDB always, ClickHouse if the sandbox is up)
uv run python -m dsbench.sftgen.dialect_conventions --reps 20 \
    --out targetA_raw.jsonl --report targetA_report.json
# render the raw rows into Qwen chat-template training JSONL (assistant-only loss, thinking on/off)
uv run python -m dsbench.sftgen.render targetA_raw.jsonl targetA_train.jsonl
```

Console scripts (`dsbench-sftgen-a`, `dsbench-sftgen-render`) are equivalent. `--dialects duckdb`
runs with no container. A row is emitted **only if** its SQL, executed against the real engine,
equals the independent pandas truth — mismatches are counted in the report, never emitted.

## Dialects

DuckDB (in-process) and ClickHouse (the ADR-003 sandbox) work out of the box. PostgreSQL and MySQL
are optional but validated — with all four, the weekday-numbering contrast spans three schemes
(execution-verified): **Sunday → ClickHouse 7 (ISO), DuckDB/Postgres 0, MySQL 1**. Bring them up as
throwaway containers and pass the drivers ephemerally (no repo/lock churn — they are generation-only):

```bash
docker run -d --name sftgen-pg -e POSTGRES_PASSWORD=sftgen -e POSTGRES_USER=sftgen \
    -e POSTGRES_DB=sftgen -p 55432:5432 postgres:16
docker run -d --name sftgen-mysql -e MYSQL_ROOT_PASSWORD=rootsftgen -e MYSQL_DATABASE=sftgen \
    -e MYSQL_USER=sftgen -e MYSQL_PASSWORD=sftgen -p 33061:3306 mysql:8

export SFTGEN_PG_DSN="host=127.0.0.1 port=55432 user=sftgen password=sftgen dbname=sftgen"
export SFTGEN_MYSQL_DSN='{"host":"127.0.0.1","port":33061,"user":"sftgen","password":"sftgen","database":"sftgen"}'
uv run --with 'psycopg[binary]' --with pymysql python -m dsbench.sftgen.dialect_conventions --reps 20 ...
```

An engine with no DSN (or an unreachable one) is simply skipped — a dialect only ever emits
execution-verified rows. Validated end-to-end: 192 rows across all four dialects, 0 rejected.

## Files

- `schema.py` — the row contract (`SFTRow`, `Turn`, `Provenance`, `Verification`) + (de)serialisation.
- `synth.py` — six seeded non-aviation domains (retail / IoT / support / payments / web / gym),
  reproducible from `(domain, seed)`.
- `conventions.py` — the trap families: `weekday-numbering`, `weekend-flag` (the `da_weekend_delay`
  gap), `timezone-direction`, `month-bucket`; each with paraphrased question variants.
- `engines.py` — dialect execution backends (DuckDB, ClickHouse, optional Postgres/MySQL).
- `dialect_conventions.py` — Target A generator + CLI + run report.
- `render.py` — raw rows → Qwen chat-template training JSONL.
- `decontaminate.py` — the dsbench-disjointness gate (run over the rendered mixture before training).
- `teacher.py` — pluggable licence-clean teacher client (HTTP OpenAI-compatible + offline stub).
- `denominator_reasoning.py` — Target B generator: execution-verified traps + teacher execution-filter.
- `ml_tasks.py` — synthetic, non-aviation ML sandbox tasks for Target C (7 families;
  oracle-gated: `selftest(reps)`).
- `ml_task_controls.py` — the negative half of that gate: the careless approach must FAIL.
- `ml_delivery_trajectories.py` — Target C generator: teacher-as-agent, keeps oracle-passing runs.
- `probe/` — the held-out generalisation probe (6 problems on fresh domains) + its runner; the
  independent validation set that proves the fine-tune generalises rather than memorising dsbench.

## Target B (teacher-hosted)

```bash
# with a licence-clean teacher served on dashi behind the SSH tunnel (Ling-3.0-flash, MIT):
uv run python -m dsbench.sftgen.denominator_reasoning --reps 40 \
    --base-url http://localhost:18080/v1 --model ling-3.0-flash-q6-mtp --out targetB_raw.jsonl
```

Four trap families, each with the data **inlined** in the prompt (a teacher with no data just
recites `SUM/COUNT`; with the data, the only thing separating right from wrong is the
denominator/population choice). The generator computes each truth two ways (pandas + DuckDB) and
keeps a teacher trace ONLY if its answer matches — teacher supplies phrasing, ground truth supplies
correctness. The teacher must be a DIFFERENT family from Qwen (Qwen/Ornith share the gap, so they
would be low-yield teachers here); Ling-3.0-flash (inclusionAI, a hybrid reasoning MoE) gave 8/8.

## Target C (teacher-as-agent)

```bash
# gate the synthetic tasks BOTH WAYS first (no teacher, but needs the sandbox), then run the
# teacher as an agent. Both gates must pass before spending teacher time:
uv run python -m dsbench.sftgen.ml_tasks --reps 10        # solvable: reference clears the bar
uv run python -m dsbench.sftgen.ml_task_controls          # failable: the careless approach loses
uv run python -m dsbench.sftgen.ml_delivery_trajectories --reps 20 \
    --base-url http://localhost:18080/v1 --model ling-3.0-flash-q6-mtp --out targetC.jsonl
```

Run the oracle with `--reps > 1`. A single rep only proves the task is solvable on ONE dataset; a
bar the reference clears by a hair there can fail on most other seeds and silently discard good
teacher trajectories at generation time (`mlc_ticket_route` failed its own oracle on 5 of 8 seeds
before retuning). `ml_task_controls` is the other half of the gate: each task also has to be
FAILABLE by the shortcut it teaches against, or a kept trajectory demonstrates nothing. That check
caught `mlc_credit_leak`'s leakage trap scoring 0.814 with the leak included — an imperfect leak
left the tree's `flag=0` node impure, so the model quietly fell back on the honest features.

Topping an existing slice up needs `--run-offset`. The run index IS the dataset seed, so a second
pass starting from 0 re-creates the same datasets and fills the pool with duplicate trajectories --
which nothing downstream catches, because the assembler decontaminates against dsbench, not against
targetC itself:

```bash
# the slice already used run indices 0..29 for these two families; continue past them
uv run python -m dsbench.sftgen.ml_delivery_trajectories --reps 2 --run-offset 30 \
    --tasks mlc_widget_defect,mlc_delivery_time \
    --base-url http://localhost:18080/v1 --model ling-3.0-flash-q6-mtp --out targetC_topup.jsonl
```

The seven families each target a different delivery failure mode: balanced classification
(`mlc_widget_defect`), regression against a baseline (`mlc_delivery_time`), rare-event ranking
(`mlc_churn_rare`), a temporal train/test boundary (`mlc_energy_load`), multiclass with a
categorical deliverable (`mlc_ticket_route`), post-outcome leakage (`mlc_credit_leak`), and
multi-table aggregation (`mlc_upsell_join`).

The teacher operates the ADR-003 sandbox (run_sql / run_python / finish) on synthetic non-aviation
ML tasks and only trajectories that **pass the oracle** (correct-schema table, every row predicted,
metric beats the bar) are kept — a kept trajectory is by construction a demonstration of *finishing*
the loop. Records are OpenAI-format messages + the tool schemas + an assistant-only loss mask.

## Held-out probe (validation)

```bash
uv run python -m dsbench.sftgen.probe.tasks          # oracle-gate the 6 probe problems (no model)
# baseline the CURRENT base now (the "before"), re-run on the fine-tune later (the "after"):
uv run python -m dsbench.sftgen.probe.runner --provider dashi-qwen36 --model qwen36 --repeat 5
```

Six problems (2 per skill) on domains that appear in **neither** dsbench nor the generators
(clinic / rideshare / grid / marketplace / loans / energy), hand-written (not generator-produced).
Run through the same pi harness with a neutral, aviation-free system prompt. The generalisation
claim is that the fine-tune moves **both** the probe and dsbench; a dsbench gain without a probe
gain is the overfitting alarm (ADR-004 kill criterion).

Generated `*.jsonl` are artifacts, not source — write them outside the repo (e.g. a scratch dir).
