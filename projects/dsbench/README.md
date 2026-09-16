# dsbench

A small, **execution-verified** benchmark for the Qwen3.x-35B-A3B coding fine-tune study (see
[`docs/adr/ADR-001-finetune-35b-coding-dashi.md`](docs/adr/ADR-001-finetune-35b-coding-dashi.md)).
It answers one question at every stage of training: *did the model get better at our work without
regressing?*

**Two generations:**
- **v1 — single-shot** (this README; [ADR-002](docs/adr/ADR-002-dsbench-harness.md)): the model
  writes one code block; the harness runs it against fixed in-memory fixtures and asserts on the
  output. The fast, deterministic regression floor.
- **v2 — agentic aviation sandbox** ([`sandbox/README.md`](sandbox/README.md);
  [ADR-003](docs/adr/ADR-003-agentic-aviation-sandbox.md)): a tool-calling agent operates a **live**
  ClickHouse warehouse of real aviation data over a `run_sql` / `run_python` / `finish` loop, graded
  on the warehouse state it produces. The real-world-facing signal the fine-tune targets.

Scope (both): **data engineering** (`de`), **data analysis** (`da`), **data science** (`ds`), each
tagged `easy` / `medium` / `hard` / `expert`. The model writes code (v1) or drives tools (v2); the
harness executes it and asserts on what it produced. No LLM judge for the core signal.

## How a problem works

One file per task under `src/dsbench/problems/<cat>/`. Each defines a `PROBLEM` with:

- `prompt` - what the model sees. It states the exact output contract (a `solve(...)` function, or
  one SQL SELECT) and the required columns/return type.
- `make_inputs()` - deterministic fixtures (fixed seeds). python mode returns the kwargs for
  `solve`; sql mode returns `{"tables": {name: DataFrame}}`.
- `check(result)` - runs in the trusted parent, rebuilds the inputs, computes the expected answer
  independently, and compares (value-multiset for DataFrames, tolerance for floats). Returns
  pass/fail (+ reason).
- `reference` - the gold solution. `dsbench-selftest` runs every reference through the real
  sandbox+checker so the checkers are trusted before any model is scored.

Two answer modes: `python` (model returns a ```python block defining `solve`; we call it) and
`sql` (model returns one SELECT; we run it in in-memory DuckDB over the fixture tables).

## Install

From the aviai repo root (shared uv workspace):

```bash
uv sync
```

## Run

```bash
# 1. Validate the benchmark itself (all reference solutions must pass). Do this after every edit.
uv run --package dsbench dsbench-selftest

# 2. Score a served model. --label names the run (base / step500 / final ...).
uv run --package dsbench dsbench-run --base-url http://dashi:8080 --model qwen3.6 --label base

# filters + pass@k
uv run --package dsbench dsbench-run --category de --difficulty hard --label de-hard
uv run --package dsbench dsbench-run --k 3 --temperature 0.6 --label base-passat3

# 3. Compare two runs as a PAIRED test (before/during/after are just runs with different labels).
uv run --package dsbench dsbench-compare reports/runs/<base>.json reports/runs/<final>.json
```

Runs are written to `reports/runs/<timestamp>-<label>.{json,md}`. The JSON keeps each raw response
and the extracted code, so a later checker fix can be re-scored offline and a failure is debuggable.

## before / during / after workflow

1. **Before:** `dsbench-run --label base` on the base model (Gate 0 in the ADR).
2. **During:** at each checkpoint serve it and `dsbench-run --label stepNNNN`; `dsbench-compare base stepNNNN`.
3. **After:** `dsbench-run --label final`; `dsbench-compare base final`.

`dsbench-compare` reports the per-cell delta, the **regressions** (were passing, now fail - the
thing "without losing performance" is about), the gains, and an exact **McNemar** p-value over the
flips. Because the set is small, trust the paired flips and the p-value, not the raw percentage: a
handful of problems is within noise. Grow the set (more problems per cell) as the fine-tune matures.

## v2 — agentic aviation sandbox

v2 replaces single-shot code with a **tool-calling agent** operating a **live** ClickHouse warehouse
of real aviation data (BTS flights + METAR/TAF + NOTAM). Bring the stack up and load the data per
[`sandbox/README.md`](sandbox/README.md); source licenses are tracked in
[`DATA_LICENSES.md`](DATA_LICENSES.md). Then:

```bash
# Validate the agentic benchmark: every reference solution must reach the graded state. After each edit.
uv run --package dsbench dsbench-agent-selftest

# Run the agent against a served model (thinking OFF by default; --think to enable).
uv run --package dsbench dsbench-agent-run --base-url http://localhost:18080 --model ornith --label base

# One problem (or --category), with a verbose per-step trace:
uv run --package dsbench dsbench-agent-run --ids ds_notam_classify --verbose
```

Grading is **state-based**: each problem seeds a clean per-problem scratch database, the agent builds
tables / returns an answer through the tools, and an independent `check()` recomputes the truth from
the raw `aviation.*` data (no self-report to game). The loop **streams** completions and reassembles
tool calls client-side, so a model that fumbles one tool call gets a recoverable error instead of
crashing the whole run — see ADR-003. Runs land in `reports/agentic-runs/<timestamp>-<label>.{json,md}`
with the full trajectory kept for offline re-grading.

## Design notes / caveats

- **Determinism:** score at `--temperature 0` (default) for a reproducible number; use `--k>1` with
  a positive temperature only for a pass@k view.
- **Robustness vs strictness:** DataFrame checks ignore column names and row order (value multiset);
  the two modeling problems (`ds_medium_01`, `ds_hard_01`) grade a property (accuracy band) instead
  of an exact number, to stay stable across scikit-learn versions.
- **Trust model:** the sandbox runs model-generated code in a subprocess with a timeout for
  robustness, not adversarial isolation. It is fine for your own model on your own box; do not point
  it at an untrusted endpoint without a container.
- **Serving parity:** the harness hits `/v1/chat/completions`, the same path the fork serves, so
  base vs fine-tune are compared through the identical stack. If you serve with the corrected Qwen3.5
  chat template, evaluate with it too.
- **Not covered yet (grow here):** dbt/Spark/Airflow-specific tasks, multi-file/repo tasks,
  tool-calling format checks, and plotting. The one-file problem format is meant to make adding
  these cheap.
