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
| **B** | `denominator_reasoning.py` | correct conditional-population / ratio denominator | **done** — 4 trap families, inline-data prompts, execution-filtered; live yield 8/8 with the Ling teacher |
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
are optional: set `SFTGEN_PG_DSN` / `SFTGEN_MYSQL_DSN` (and install `psycopg` / `pymysql`) and they
join automatically — otherwise they are skipped. The weekday-numbering contrast is sharpest with
MySQL present (Sunday=1), but ClickHouse ISO (Sunday=7) vs DuckDB/Postgres (Sunday=0) already
exercises the trap.

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
- `ml_tasks.py` — synthetic, non-aviation ML sandbox tasks for Target C (oracle-gated: `selftest()`).
- `ml_delivery_trajectories.py` — Target C generator: teacher-as-agent, keeps oracle-passing runs.

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
# oracle-gate the synthetic tasks first (no model), then run the teacher as a sandbox agent:
uv run python -m dsbench.sftgen.ml_tasks           # setup -> reference -> check, must PASS
uv run python -m dsbench.sftgen.ml_delivery_trajectories --reps 20 \
    --base-url http://localhost:18080/v1 --model ling-3.0-flash-q6-mtp --out targetC.jsonl
```

The teacher operates the ADR-003 sandbox (run_sql / run_python / finish) on synthetic non-aviation
ML tasks and only trajectories that **pass the oracle** (correct-schema table, every row predicted,
metric beats the bar) are kept — a kept trajectory is by construction a demonstration of *finishing*
the loop. Records are OpenAI-format messages + the tool schemas + an assistant-only loss mask.

Generated `*.jsonl` are artifacts, not source — write them outside the repo (e.g. a scratch dir).
