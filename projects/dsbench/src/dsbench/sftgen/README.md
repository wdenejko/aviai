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
| **A** | `dialect_conventions.py` | SQL-dialect date/time conventions (weekday numbering, tz direction) | **done** — execution-verified, teacher-free |
| **B** | `denominator_reasoning.py` | correct conditional-population / ratio denominator | todo |
| **C** | `ml_delivery_trajectories.py` | agentic ML-workflow delivery discipline | todo |
| — | `decontaminate.py` | 13-gram + schema-identifier + numeric-answer gate vs dsbench | todo |

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
- `synth.py` — seeded non-aviation domains (retail / IoT / support), reproducible from `(domain, seed)`.
- `conventions.py` — the trap families: `weekday-numbering`, `timezone-direction`.
- `engines.py` — dialect execution backends (DuckDB, ClickHouse, optional Postgres/MySQL).
- `dialect_conventions.py` — Target A generator + CLI + run report.
- `render.py` — raw rows → Qwen chat-template training JSONL.

Generated `*.jsonl` are artifacts, not source — write them outside the repo (e.g. a scratch dir).
