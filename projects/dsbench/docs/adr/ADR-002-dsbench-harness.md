# ADR-002: dsbench - an execution-verified DE / DA / DS benchmark for the fine-tune study

- **Status:** Accepted - reference implementation built and validated 2026-09-15
- **Date:** 2026-09-15
- **Deciders:** Wojtek Denejko (box owner)
- **Relates to:** ADR-001 (the fine-tuning plan whose gates this benchmark scores); the avtext harness (`projects/avtext/src/avtext/harness/models.py`), whose OpenAI-over-HTTP seam this reuses
- **Implementation:** `projects/dsbench/` in this repo. The harness and an 18-problem seed set are built, self-validated (`dsbench-selftest` 18/18) and lint-clean. This ADR documents the design as built and the backlog. The next agent EXTENDS this code; it does not rebuild it.

## Context

ADR-001 concludes that dashi can LoRA-fine-tune the 35B-A3B MoE for data-science / data-engineering / software-engineering work, and that "without losing performance" splits into inference speed (preserved by construction if we merge and re-quantize to the same GGUF recipe) and general quality (must be measured). The whole plan is gated on before/after measurement, so it needs an evaluation instrument that is:

- **Owned and tailored.** Public DS/DE benchmarks (DS-1000, DABStep, DA-Code, BIRD, Spider) are valuable and used as external checks (ADR-001 Appendix B.3), but they are large, mostly agentic scaffolds, and not shaped to the owner's stack (pandas/polars, SQL/dbt/Spark/Airflow, warehouse work). A small, house benchmark we control gives a fast, cheap signal we can iterate on and grow toward the exact skills we care about.
- **Execution-verified.** The recurring lesson from the research (ADR-001 Appendix B.5) is that imitation SFT saturates as the base model gets stronger, so gains are small and must be measured precisely. Grading generated code by running it and asserting on the output is the only signal that stays honest across model iterations. LLM-judge grading is explicitly avoided for the core score.
- **Paired and noise-aware.** A small set has a few points of run-to-run noise (a ~20-problem set has ~1 sigma ~= 2-3 problems; ADR-001 Appendix B.3). So the instrument reports paired per-problem flips and an exact McNemar p-value, not just a percentage, and the "without losing performance" question is answered by the list of regressions (problems that were passing and now fail).
- **Reusable at every stage.** before / during (per checkpoint) / after are the same run with a different label; comparison is between any two runs.

## Decision

Build a self-contained aviai project, `dsbench`, that:

1. Represents each task as one small self-contained Python module: deterministic fixtures, the prompt shown to the model, an independent checker, and a gold reference solution.
2. Supports two answer modes: `python` (the model returns a `solve(...)` function that is executed) and `sql` (the model returns one SELECT executed in in-memory DuckDB).
3. Runs the model's code in a throwaway subprocess with a timeout; grades the pickled result in the trusted parent.
4. Reaches the model over an OpenAI-compatible HTTP endpoint (the same path the fork serves), so base vs fine-tune are compared through an identical stack.
5. Validates its own checkers first: `dsbench-selftest` runs every gold reference through the real sandbox and checker and must pass 100% before any model is scored (the oracle gate, in the spirit of avtext's oracle-first methodology).
6. Tags every problem by category (de/da/ds) and difficulty (easy/medium/hard) and reports the score as a category x difficulty table plus a paired before/after comparison.

## Implementation details (as built)

### Location and layout

```
projects/dsbench/
  pyproject.toml                 # hatchling package, workspace member; console scripts
  README.md                      # run instructions + design notes
  docs/adr/                      # ADR-001 (context), ADR-002 (this)
  reports/runs/                  # <timestamp>-<label>.{json,md} run outputs
  src/dsbench/
    schema.py                    # Problem dataclass + ProblemResult; the one contract
    loader.py                    # discover problems/<cat>/*.py -> [Problem]
    problems/
      de/  da/  ds/              # one module per problem, each exposes PROBLEM
    harness/
      client.py                  # OpenAI-compatible chat over httpx
      extract.py                 # strip <think>, take last fenced block, fallback
      sandbox.py                 # run candidate in subprocess, grade in parent
      _worker.py                 # subprocess entry: run one candidate vs one problem
      checks.py                  # comparators (value-multiset, tolerance, dict, frame)
      score.py                   # aggregate -> category x difficulty markdown table
      runner.py                  # dsbench-run  (score a served model)
      compare.py                 # dsbench-compare  (paired before/after + McNemar)
      selftest.py                # dsbench-selftest (validate all references)
```

### Problem contract (`schema.Problem`)

A frozen dataclass, one instance per module named `PROBLEM`:

- `id` (e.g. `de_hard_01`), `category` in {de, da, ds}, `difficulty` in {easy, medium, hard}, `title`, `tags`.
- `prompt`: the exact instruction shown to the model. It states the output contract precisely: for python mode "return a single ```python code block defining `solve(...)` that returns X"; for sql mode "return one ```sql SELECT; the tables are ...". Ambiguity in the prompt is the main way a checker unfairly fails a correct model, so prompts fix the return type and, where they matter, the column names.
- `mode`: `python` or `sql`.
- `make_inputs() -> dict`: deterministic fixtures (fixed seeds, fixed data). python mode returns the kwargs passed to the model's function; sql mode returns `{"tables": {name: DataFrame}}`. It MUST be deterministic, because the checker calls it again to compute the expected answer.
- `check(result) -> bool | (bool, reason)`: runs in the trusted parent. It rebuilds the inputs via `make_inputs()`, computes the expected answer INDEPENDENTLY (a reference computation, not by trusting the model), and compares. Returning a reason makes failures self-explaining.
- `reference`: the gold solution as the exact string a perfect model would emit (a python function or a SQL query). Powers `dsbench-selftest`. Also documents the intended answer for whoever grows the set.
- `entrypoint` (python mode, default `solve`), `timeout` (default 25 s).

### Answer modes

- **python:** the worker execs the model's code in a namespace preloaded with `pd`/`np` (notebooks always have them; the model may still import its own), fetches `PROBLEM.entrypoint`, and calls it with the `make_inputs()` kwargs. The return value is graded.
- **sql:** the worker registers each `make_inputs()["tables"]` DataFrame into an in-memory DuckDB connection and runs the model's query; the result DataFrame is graded. DuckDB is the SQL engine so text-to-SQL is gradable without a warehouse.

### Code extraction (`extract.py`)

Strip `<think>...</think>` (Qwen thinking mode) first, then take the LAST fenced block whose language matches the mode (last, because models often show a wrong first attempt then a corrected one). Fallbacks: any last fenced block; for SQL, salvage from the first `SELECT`/`WITH`; else the trimmed message. This tolerates prose-wrapped answers without being lax about which block is the answer.

### Sandbox and isolation (`sandbox.py` + `_worker.py`)

The model's code runs in a separate process (`python -m dsbench.harness._worker <module> <code_file> <result_file>`), so a crash, an infinite loop, or a memory blow-up is contained; the parent enforces `problem.timeout` and kills a hung worker. Only the pickled result crosses back. Statuses: `ok` (checker accepted), `wrong` (ran, incorrect), `error` (raised; last traceback line kept), `timeout`, `no_code` (nothing extracted). The checker runs in the parent and is itself guarded so a buggy checker fails loud (`error: checker raised: ...`) rather than silently passing.

**Trust model:** the subprocess is for robustness, not adversarial isolation. It is fine for your own model on your own box. Do not point this harness at an untrusted endpoint without a container/seccomp. This is stated in the worker docstring and README.

### Checkers (`checks.py`)

Grading DS/DE output fails in two directions: too strict (float noise, column aliasing, row order) or too loose (right shape, wrong values). The helpers pick middles:

- `values_equal(got, expected)`: DataFrames compared as a value MULTISET - same number of columns, same rows ignoring column names and row order. Robust to SQL aliasing and column reordering. Floats rounded (default 6 dp); every flavour of missing (NaN, None, pd.NA, NaT) normalized to one sentinel so missing == missing (Python's `NaN != NaN` bit us during self-test and is fixed here).
- `frame_equal(got, expected)`: stricter, column names must match (case-insensitive), used when the prompt fixes output columns.
- `approx(a, b, rtol, atol)`: scalar closeness with NaN == NaN, for correlations/coefficients/metrics.
- `dicts_approx(got, expected)`: same keys, numeric values close, for "return a dict of group -> metric".

Two ML modeling problems (`ds_medium_01`, `ds_hard_01`) grade a PROPERTY (an accuracy band) rather than an exact number, so the score does not drift with scikit-learn versions. This is a deliberate trade: it accepts any leakage-free, correct approach and stays reproducible, at the cost of not distinguishing two correct-but-different pipelines.

### Model client (`client.py`)

`ChatClient` POSTs to `/v1/chat/completions` via httpx: `base_url` (default `http://127.0.0.1:8080`), `model`, `temperature` (default 0 for a reproducible score), `top_p`, `max_tokens` (default 2048), optional `system`. No inference library is a dependency, exactly like avtext, so the served fine-tune plugs into the same path as the base model. One client per worker thread (httpx.Client is not thread-safe to share).

### Runner (`dsbench-run`)

Loads and filters problems (`--category`, `--difficulty`, `--ids`, `--limit`), runs them concurrently (`--concurrency`, default 4) with `pass@k` (`--k`, default 1; k>1 wants `--temperature`>0). Per problem: call the model, extract code, sandbox+check, record a `ProblemResult` keeping status, reason, the extracted code and the raw response (so a failure is debuggable and a later checker fix can be re-scored offline). Writes `reports/runs/<timestamp>-<label>.json` and a markdown summary; prints the category x difficulty table.

### Score (`score.py`)

Aggregates results into the (category x difficulty) -> (passed, total) table, overall pass rate, and a list of failures with status and reason (a run is meant to be actionable, not just a number).

### Compare (`dsbench-compare`)

Loads two run JSONs, pairs by problem id, and reports: overall before -> after with delta; gains (fail -> pass) and regressions (pass -> fail) listed by id; the per-cell before -> after table; and an EXACT two-sided McNemar p over the discordant pairs. On a small set, trust the paired flips and the p-value, not the raw percentage.

### Selftest (`dsbench-selftest`)

Runs every problem's `reference` through the real sandbox + checker and asserts all pass. If a reference fails, the bug is in the problem (fixtures/checker/gold), not the model. It is the gate that must be green before trusting any score, and it belongs in CI. Current status: 18/18.

### Dependencies and workspace

`dsbench` is a uv-workspace member (hatchling, `src/dsbench` layout). Unlike avtext (which only parses text), this benchmark EXECUTES model code, so the data stack is a runtime dependency: `httpx, pandas, numpy, polars, duckdb, pyarrow, scikit-learn, scipy`. These were added to the shared aviai venv (`uv sync`; `uv.lock` changed). Console scripts: `dsbench-run`, `dsbench-compare`, `dsbench-selftest`. Run with `uv run --package dsbench <script>`.

## The seed set (18 problems)

Six per category, two per difficulty. Each is one file under `problems/<cat>/`.

| id | title | mode |
|---|---|---|
| de_easy_01 | Revenue per category | sql |
| de_easy_02 | Latest record per user | python |
| de_medium_01 | Top 2 products per category | sql |
| de_medium_02 | Clean a messy extract | python |
| de_hard_01 | Sessionize by 30-min gap | sql |
| de_hard_02 | Upsert / SCD merge | python |
| da_easy_01 | Mean score per group | python |
| da_easy_02 | Count with a filter | python |
| da_medium_01 | Conversion rate per cohort | python |
| da_medium_02 | Revenue per country (join) | python |
| da_hard_01 | Weekly rolling mean | python |
| da_hard_02 | Cohort month-1 retention | python |
| ds_easy_01 | Pearson correlation | python |
| ds_easy_02 | Linear regression slope | python |
| ds_medium_01 | Train/test classification accuracy | python |
| ds_medium_02 | Precision / recall / F1 | python |
| ds_hard_01 | Leakage-free CV pipeline | python |
| ds_hard_02 | ROC AUC | python |

## Validation performed (2026-09-15)

- `dsbench-selftest`: 18/18 references pass through the real sandbox and checkers.
- Negative controls (must fail, and do): a wrong python answer -> `wrong`; a wrong SQL aggregate -> `wrong`; a returned constant for a float metric -> `wrong`; a division by zero -> `error`; an empty response -> `no_code`. Confirms the checkers discriminate rather than accept everything.
- Fenced-extraction path: wrapping each gold reference in a Markdown code fence (as a model would) and running it through `extract_code` + sandbox + checker passes 18/18.
- `dsbench-compare`: smoke-tested on two synthetic runs (paired deltas, gains/regressions lists, McNemar p rendered).
- `ruff check` clean under the repo config.
- One real bug was caught by the self-test and fixed: `values_equal` compared row tuples with `==`, so any expected result containing a missing value never matched (`de_medium_02`). The comparator now normalizes missing values.

## Options considered

### Execution-verified vs LLM-judge grading - **execution chosen**
Running the code and asserting on output is objective and stable across model iterations; an LLM judge is noisy, needs a fixed judge model for reproducibility, and can be gamed. Cost: authoring an execution checker per problem is more work than a judge prompt. Accepted. A judge may later be added only for genuinely open-ended items (e.g. "explain the approach"), never for the core score.

### Subprocess sandbox vs in-process exec - **subprocess chosen**
In-process is simpler but an infinite loop or a segfault in model code takes down the whole run, and there is no clean timeout. The subprocess boundary costs a process spawn per attempt (tens of ms, negligible next to model latency) and buys containment and a hard timeout. Accepted.

### DataFrame equality: value-multiset vs exact frame - **value-multiset default**
Models legitimately alias columns and return rows in any order, especially for SQL. Comparing the value multiset (ignoring names and order) avoids unfair failures; `frame_equal` is available when the prompt fixes column names and they are part of correctness. Risk: value-multiset can accept a transposed/mislabeled result in rare shapes; mitigated by fixing the return contract in the prompt.

### Exact number vs property for ML problems - **property for the two modeling problems**
An exact CV-score/accuracy match is brittle across scikit-learn versions. Grading a band (e.g. accuracy in [0.9, 1.0] on separable data) is reproducible and accepts any correct approach. Trade: it will not distinguish two correct pipelines or catch a subtle leak that still scores in-band. Accepted for the seed; a stricter, version-pinned variant can be added later.

### HTTP client vs in-process inference - **HTTP chosen**
Matches avtext and keeps inference libraries out of the package (CI stays light, Mac/Linux both fine), and guarantees the fine-tune is evaluated through the exact serving path. Cost: needs a running server. Accepted.

### One file per problem vs a single data file - **one file per problem**
A module per problem keeps fixtures, prompt, checker and gold together and readable, and makes "add a problem" a one-file change with no registry to edit. Cost: slightly more boilerplate than JSON rows. Accepted; it is what makes the set cheap to grow, which is the whole point.

## Consequences

- A fast, honest, house signal for the fine-tune loop: before / during / after are three `dsbench-run` calls; `dsbench-compare` names the regressions.
- The harness is generic: any OpenAI-compatible endpoint, any of the three categories, any difficulty. It also measures the no-training alternatives from ADR-001 (better prompts, retrieval, other community tunes) on the same axis.
- New maintenance surface: the data stack now lives in the shared venv; a problem author must keep `make_inputs` deterministic and the checker independent, and must run `dsbench-selftest` after any edit.
- The seed set is deliberately small, so a single-run percentage is within noise; conclusions come from paired flips and from growing the set per cell.

## Backlog / action items for the next agent

Ordered by leverage. Extend the existing code; keep `dsbench-selftest` green after every change.

1. [ ] **Grow the set per cell** to ~5-8 problems per (category, difficulty) so the paired test has power. Target the owner's real stack: polars alongside pandas, window/CTE-heavy SQL, joins/upserts, cleaning, cohort/funnel analysis, feature engineering, model evaluation.
2. [ ] **Data-engineering breadth the seed lacks:** dbt (model SQL + `dbt parse` / schema-test reasoning), PySpark transforms (gradable via a local Spark or by translating to DuckDB semantics), Airflow/Dagster DAG-shape questions (dependency ordering, idempotency), warehouse-dialect SQL. Some of these need a checker beyond DuckDB - decide per task whether to execute or to check structure.
3. [ ] **Tool-call / format tests.** ADR-001 flags the Qwen3.5/3.6 chat-template tool-call bug and that agent harnesses depend on the format. Add problems (or a separate mode) that assert the model emits a well-formed tool call (Qwen3-Coder XML or Hermes JSON, per the served template). This guards the "without losing performance" tool-use dimension.
4. [ ] **Contamination guard.** Before adopting any external items, 13-gram-decontaminate problems against the training mixture (ADR-001 Appendix B.4) and never source a problem from a set the model trains on. Keep the benchmark strictly held out.
5. [ ] **Serving-parity option.** Add a `/completion` (raw, pre-templated) path like avtext's `completion_predictor`, for when a hard fine-tune overfits the exact training wrapper and the chat template drifts (minja vs HF). Also add a `--chat-template` note so eval uses the same corrected template the fork serves.
6. [ ] **pass@k and sampling.** The runner supports `--k`; add a fixed-seed sampling story and, if wanted, an `avg@k` metric for temperature>0 runs, alongside the default deterministic pass@1.
7. [ ] **Reporting.** An HTML/one-page report (avtext has a dashboard pattern) and a small `reports/` index across runs, so before/during/after is visible at a glance.
8. [ ] **Wire into the ADR-001 gates.** Gate 0 baseline (Qwen3.6-35B-A3B and Ornith-1.5), Gate 2 before/after with the regression thresholds ADR-001 sets (<=1 pt on general subsets, <=2 pt on general-coding, plus MTP acceptance which is measured separately, not here).
9. [ ] **Optional stricter modeling checks.** Version-pin scikit-learn and add exact-number variants of the two property-graded problems, and a deliberate leakage-trap problem where a leaky pipeline scores detectably higher.
10. [ ] **CI.** Run `dsbench-selftest` + `ruff` in the aviai CI so a broken problem is caught before it is trusted.

## Risks and caveats

- **Small-set noise:** ~2-3 problems is ~1 sigma on the seed; do not act on a sub-6-point single-run delta - use paired McNemar and grow the set (backlog 1).
- **Trust model:** runs model code in a subprocess, not a hardened sandbox; own model, own box only.
- **Checker fairness:** a too-strict checker fails a correct model. The prompt must fix the return contract; `dsbench-selftest` only proves the gold passes, not that every correct alternative does - watch `wrong` statuses on plausible answers when adding problems.
- **sklearn/pandas drift:** property checks absorb version drift for the two ML problems; exact-number problems (metrics, regression slope) rely on stable library behaviour - re-run `dsbench-selftest` after any dependency bump.
- **Serving template:** evaluate with the same (corrected) chat template the fork serves, or a hard fine-tune can fall off distribution and score unfairly low (ADR-001 template notes; backlog 5).
