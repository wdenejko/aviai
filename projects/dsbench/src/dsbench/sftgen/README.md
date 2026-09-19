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
| **B** | `denominator_reasoning.py` | correct conditional-population / ratio denominator | **generator done** — teacher-agnostic + execution-filter verified offline; needs a hosted teacher to produce rows |
| **C** | `ml_delivery_trajectories.py` | agentic ML-workflow delivery discipline | todo (needs teacher) |
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

## Target B (needs a hosted teacher)

```bash
# with a licence-clean teacher served on dashi behind the SSH tunnel (e.g. gpt-oss-120b):
uv run python -m dsbench.sftgen.denominator_reasoning --reps 40 \
    --base-url http://localhost:18080/v1 --model gpt-oss --out targetB_raw.jsonl
```

The generator computes each ratio's truth two ways (pandas + DuckDB) and keeps a teacher trace ONLY
if its answer matches that truth — teacher supplies phrasing, ground truth supplies correctness.

Generated `*.jsonl` are artifacts, not source — write them outside the repo (e.g. a scratch dir).
