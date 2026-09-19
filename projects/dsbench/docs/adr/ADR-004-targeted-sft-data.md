# ADR-004: Targeted SFT data for the Qwen3.6-35B-A3B fine-tune (the three measured dsbench gaps)

- **Status:** Proposed (data-design gate for ADR-001 Gate 1/Gate 2)
- **Date:** 2026-09-19
- **Deciders:** Wojtek Denejko (box owner)
- **Relates to:** ADR-001 (the fine-tune plan — this ADR instantiates its Gate 1 "pilot mixture" and Gate 2 "targeted slice", and refines Appendix B.4 for that slice only); ADR-003 (the agentic sandbox whose oracle both *measured* these gaps and will *generate + verify* the data); memory `reference-ornith-agentic-behavior` (the Gate 0 measurement this ADR acts on).
- **Note:** ADR-001 Appendix B.4 already specifies the **breadth + replay** buckets (jupyter-agent, OmniSQL, SWE-Swiss, SmolTalk2, …). This ADR specifies the **targeted slice** — the part of the mixture chosen specifically to move the three needles Gate 0 measured — and the honesty machinery (decontamination + held-out probe + provenance) that makes a before/after on dsbench a *valid* claim rather than teaching-to-the-test.

## Question

> Gate 0 (ADR-001) measured the base. What data, generated how, will move the three measured gaps **without** overfitting to the benchmark that measures them, and **without** tainting the redistributable model's provenance?

## Context

### What Gate 0 actually measured (2026-09-19)

Qwen3.6-35B-A3B vs Ornith-1.5, same pi harness @ 0.84.4, thinking high, temp 0, k=5, both Q8_0: a **statistical tie** (Qwen 79/115 · 15/23 · Ornith 75/115 · 16/23), Qwen ahead on the target DS/DE domains (38/50 vs 34/50). The decisive finding for *this* ADR is that the majority-failures are **not noise and not Ornith's RL artifact** — both models fail the same problems, two of them with two *different* wrong answers, which is the signature of a genuine, family-wide capability gap. They cluster into exactly three movable targets, and I re-read the problem code to state each mechanically (not from the score table):

| # | Target (transferable skill) | dsbench evidence | The exact error |
|---|---|---|---|
| **A** | **SQL-dialect date/time conventions** | `da_cancel_dow` 0–2/5, `da_weekend_delay` 0–1/5, `da_utc_peak_hour` 1/5 | ClickHouse `toDayOfWeek`/`dayOfWeek` is ISO **1=Monday..7=Sunday**; the model uses another engine's numbering. And time-zone **direction**: `UTC = local − tz_offset` with US offsets negative, so ATL (UTC−4) ⇒ `local + 4`; both models add the signed offset the wrong way and land on hour **5** instead of **13** — *even though `da_utc_peak_hour`'s prompt spells the offsets out.* So it is a **directional-reasoning** miss, not a missing fact. |
| **B** | **Correct conditional population / denominator for a ratio** | `da_delay_attribution` 0/5, both models wrong *every* run (Qwen 40.9, Ornith 42.8 vs **44.1**) | The five BTS cause fields (`CarrierDelay`…`LateAircraftDelay`) are populated **only** for arrivals delayed ≥15 min (else null); the "share" denominator is the **sum of the five causes**, not total `ArrDelayMinutes` and not a per-flight average. The model reasons over the wrong base/null-semantics. Two different stable wrong answers ⇒ a real reasoning gap, not an oracle bug. |
| **C** | **Agentic ML-workflow delivery discipline** | `ds_delay_predict` / `ds_cancel_predict` / `ds_taxi_regression` all 1–4/5, high variance | The model is **competent at the ML** (a failed `ds_taxi_regression` run had tuned a GBR to MAE 6.75 < the 7.64 baseline) but fails to **deliver the artifact**: it over-tunes, or ends its turn before writing the *exact-schema* output table (`taxi_pred(id UInt32, pred Float64)`, one row per test id) to the scratch DB. `ds_notam_classify` 4–5/5 proves the write path works *when the model commits the artifact*. This is the direct descendant of the old "structured-output reliability" theme, now on real ML tasks. |

Everything else — cross-source joins, TAF dedup, fact tables, rank leaderboards, NOTAM CV — already passes by majority. So the fine-tune's job is **narrow and known**: three behaviours, all in the DS/DE wheelhouse the box can actually train (ADR-001 §Decision), none of them requiring the RL-at-scale that produced the big SWE-bench jumps.

### The honesty problem (why this ADR is mostly about decontamination, not volume)

dsbench is the **before/after metric** for the whole project. If we generate SFT data that fixes these exact problems on the exact aviation schema, a post-fine-tune score gain proves nothing — it could be memorisation of the benchmark. Three rules make the gain a *real* claim:

1. **Teach the transferable skill on a different distribution.** The data teaches "be dialect-aware about weekday numbering" and "pick the conditional-population denominator" on **non-aviation schemas, multiple SQL dialects, and different phrasings** — never on `aviation.flights` and never on the 23 problem prompts.
2. **Hard-decontaminate against dsbench** (protocol below): no training row may share substantial n-gram overlap with any problem prompt, the aviation schema identifiers, or the specific numeric answers.
3. **Validate on an independent held-out probe**, not just on dsbench. A fresh, small "convention/denominator/delivery" probe on schemas and domains the training data never touched is the real generalisation test; dsbench is the *secondary* read. (Overfitting to dsbench would move dsbench but not the probe.)

### The lever we have that most fine-tunes don't

All three targets are **execution-verifiable**: a weekday-numbering answer, a ratio denominator, and an ML output table can each be *checked by running code against a real engine* — the same oracle discipline ADR-003's sandbox already uses to grade the model. That single fact drives most of the design decisions below:

- **Provenance can be Apache-2.0-clean by construction.** The convention/denominator data is *own-generated* (our templates, run against real ClickHouse/DuckDB/Postgres/MySQL), so there is no teacher and no licence question at all. Where a natural-language *reasoning trace* is wanted, the teacher is a licence-clean model (DeepSeek/Qwen, per ADR-001 B.4) **and** every trace is execution-filtered — a trace whose final answer ≠ the verified truth is dropped. This is stronger than ordinary distillation: the label is ground truth, the teacher only supplies phrasing.
- **Quality is guaranteed, not hoped.** Every targeted row carries an execution proof. No "looks right" data enters the mixture.

## Decision — the targeted-slice spec

Three generators, one per target. Each is defined by: the **skill statement** (what the model must learn), the **source distribution** (deliberately not aviation), the **generation + verification** method, the **format**, and the **decontamination + hold-out** stance. All rows render in the corrected Qwen3.5/3.6 chat template (ChatML, tool calls in the **XML** form the shipped template uses, assistant-only loss; thinking on for reasoning rows, empty `<think></think>` for direct rows — ADR-001 B.4).

### Target A — SQL-dialect date/time conventions  *(generator: `sftgen/dialect_conventions.py`)*

- **Skill:** given a schema and a question, (i) use the **correct** date/time convention **for the stated engine**, and (ii) when the engine is ambiguous or the convention bites, **state the assumption and verify it** rather than guessing. The generalisation target is *dialect-awareness*, because the conventions genuinely differ and that difference is the trap:

  | Convention | ClickHouse | Postgres | MySQL | DuckDB |
  |---|---|---|---|---|
  | weekday number | `toDayOfWeek` **1=Mon..7=Sun** (ISO) | `EXTRACT(DOW)` **0=Sun**, `ISODOW` 1=Mon | `DAYOFWEEK` **1=Sun..7=Sat** | `dayofweek` **0=Sun** |
  | local→UTC | `UTC = local − offset`; US offsets negative ⇒ `local + |offset|` (direction is the miss) | same arithmetic, `AT TIME ZONE` semantics differ | — | — |
  | month/quarter, week-start, `toStartOf*` vs `date_trunc` | engine-specific names, same intent | | | |

- **Source distribution (NOT aviation):** small synthetic schemas in unrelated domains — retail orders, IoT sensor readings, a payments ledger, a support-ticket log, a gym-attendance table. Each generated with a seeded faker so the *data* is ours and reproducible.
- **Generation + verification:** for each (schema, domain, dialect, convention) tuple, template a natural-language question, the **reference SQL in that dialect**, and the answer computed **twice** — once by executing the SQL against a real engine instance (ClickHouse in the sandbox; Postgres/MySQL/DuckDB in throwaway containers) and once by an independent pandas re-derivation (the ADR-003 `check`/`reference` cross-validation pattern). A row is kept only if the two agree. The assistant turn shows the correct SQL and, for the ~40% "thinking" rows, a short trace that *names the convention and its numbering* ("ClickHouse `toDayOfWeek` is ISO, 1=Monday, so Thursday = 4"), which is exactly the reasoning the model currently skips.
- **Format / volume:** mostly single-turn text-to-SQL + answer; ~40% carry a thinking trace, 60% direct. **~0.5M tokens** (~1,500–2,000 rows), spread across the four dialects and ~six domains so no single schema dominates.

### Target B — conditional-population / denominator reasoning  *(generator: `sftgen/denominator_reasoning.py`)*

- **Skill:** identify the **correct population and denominator** for a rate/share/attribution question — especially when a field is populated only under a condition (nulls that mean "not applicable", not "zero"), and when "share of X across categories" means *divide by the summed categories*, not by a grand total or an average-of-averages.
- **Source distribution (NOT aviation):** the same synthetic domains as Target A, but shaped to reproduce the trap families: (i) conditional-null columns (a `refund_reason` populated only for returned orders; a `fault_code` only for failed sensor reads); (ii) share-of-total vs share-of-subtotal; (iii) pooled rate vs mean-of-per-group-rates (the `da_weighted_ontime` trap, generalised); (iv) rate with a cancelled/excluded denominator (the `da_all_flights_avg_delay` trap, generalised).
- **Generation + verification:** template the question + the **execution-verified truth** (SQL and pandas agree). Then — because the *reasoning* is the point — generate a thinking trace with a **licence-clean teacher** that must arrive at the verified number; **drop any trace whose answer ≠ truth** (execution-filtered distillation). The kept trace explicitly discusses *which rows are in scope* and *what the denominator is and why*. This is the one target where a teacher earns its keep (natural denominator-reasoning prose is hard to template well), and it stays provenance-clean because the teacher is licence-clean and the label is ground truth.
- **Implementation notes (2026-09-19).** Two decisions emerged in build: (1) **the data is inlined** in each prompt as a small markdown table — a teacher given only a schema recites `SUM/COUNT` instead of a number, so it can't be execution-filtered; with the data small enough to reason over by hand, the *only* thing separating right from wrong is the denominator/population choice, which isolates the measured skill. (2) **The teacher must be a different family from Qwen.** Qwen3.6 and Ornith both fail `da_delay_attribution` every run, so they are low-yield teachers here; DeepSeek was off-disk, so the hosted teacher is **`Ling-3.0-flash`** (inclusionAI, MIT, a 124B/5.1B-active hybrid reasoning MoE), served on dashi via the proven `toolbox` path with `--reasoning-format deepseek`. Live smoke: **8/8** kept, and it does the pooled sum correctly rather than falling into the mean-of-means trap. Four trap families are implemented: `conditional-null-share`, `pooled-vs-mean-rate`, `share-of-subtotal`, `excluded-denominator`.
- **Format / volume:** ~80% thinking rows (the reasoning is the skill), 20% direct. **~0.6M tokens** (~1,200–1,600 rows) across the four trap families.

### Target C — agentic ML-workflow delivery discipline  *(generator: `sftgen/ml_delivery_trajectories.py`)*

- **Skill:** run the full agentic ML loop in a sandbox and **finish it** — inspect the tables, train a reasonable model, **write predictions for EVERY test row to the exact table name and schema stated in the prompt**, then **stop** (no over-tuning past a good-enough bar). The failure being trained out is "competent model, no delivered artifact."
- **Source distribution (NOT the dsbench aviation tasks):** dsbench-*shaped* but on different public datasets loaded into the sandbox (e.g. UCI/OpenML tabular sets — bike-sharing demand, adult-income, telco-churn, a house-price regression), each wrapped in the ADR-003 harness with an oracle that (a) states the exact output-table contract and (b) grades by a held-out metric with a beat-the-trivial-baseline bar — mirroring `ds_*` **mechanics** without reusing their **content**.
- **Generation + verification:** run a **licence-clean teacher** (DeepSeek/Qwen) as the agent inside the sandbox on these held-out tasks; **keep only trajectories that pass the oracle** (correct-schema table, every row predicted, metric beats baseline). The kept trajectory — tool calls, `run_python`, the `CREATE TABLE … ENGINE=MergeTree ORDER BY …` + `insert_df`, and the terminal "done" — becomes a multi-turn training sample. This is execution-verified by definition (a trajectory only exists as data if it *delivered*), which is precisely the behaviour to reinforce. Include a few "recovered" trajectories (model hits an error, fixes it, still delivers) since ADR-003 notes the model *can* self-recover when it commits.
- **Format / volume:** multi-turn tool-call trajectories, thinking on, assistant-only loss on the assistant/tool-call turns; tool results as `<tool_response>` user turns (B.4). Trajectories are long, so **~0.9M tokens** is only ~150–250 trajectories — generation cost is the teacher inference, hours on the box or a few dollars of API (ADR-001 B.4 cost model).

### The targeted slice inside ADR-001's mixtures

The targeted slice is **~2M tokens (~20%)** of the Gate 2 10M-token first run; the rest stays as ADR-001 B.4 breadth + replay. Rationale: three narrow behaviours must not crowd out breadth (or the model over-specialises and regresses — the very thing ADR-001 Gate 2's thresholds guard). Replay stays at 25–30%.

| Bucket | Source | Tokens | % of 10M |
|---|---|---|---|
| **Target A — dialect conventions** | `sftgen/dialect_conventions.py` (own-gen, exec-verified) | 0.5M | 5 |
| **Target B — denominator reasoning** | `sftgen/denominator_reasoning.py` (teacher trace, exec-filtered) | 0.6M | 6 |
| **Target C — ML-delivery trajectories** | `sftgen/ml_delivery_trajectories.py` (teacher-in-sandbox, oracle-passed) | 0.9M | 9 |
| Breadth (DS/DE/SWE/general code) | ADR-001 B.4 buckets | 5.5M | 55 |
| General replay | ADR-001 B.4 (SmolTalk2/Dolci/Tulu-3) | 2.5M | 25 |

The **Gate 1 pilot (1M tokens)** is a proportional down-sample of this table: ~0.2M targeted (all three generators represented) + ~0.8M breadth/replay — enough to prove the pipeline (render → decontaminate → train → convert → serve → parity) end-to-end before spending a Gate 2 GPU window.

## Provenance & licensing

- **Targets A and the *labels* of B/C are own-generated + execution-verified ⇒ Apache-2.0-clean by construction** (no teacher). This is the bulk of the redistributable value.
- **Teacher-authored text (B reasoning traces, C trajectory prose) uses only licence-clean teachers** — DeepSeek (MIT), Qwen (Apache-2.0), GLM/Kimi/gpt-oss — per ADR-001 B.4 and its taint list. No Claude/OpenAI/Gemini teacher output enters anything redistributable.
- **Every teacher row is execution-filtered**, so even the distilled portion is anchored to ground truth, not to the teacher's opinion.
- Record the teacher, model version, and verification method per row (a `provenance` field) so the data-licence audit (ADR-001 Gate 3 publish gate) is mechanical.

## Decontamination protocol (dsbench-specific, automated)

A row is **rejected** from the targeted slice (and flagged in breadth/replay) if any of:

1. **13-gram overlap** ≥ 1 shared 13-gram with any of the 23 problem `prompt` strings (ADR-001 B.4 uses 13-gram against external eval sets; we add dsbench's own prompts).
2. **Schema-identifier hit:** contains `aviation.flights`, the METAR/TAF/NOTAM table names, or the distinctive column identifiers (`CRSDepTime`, `ArrDelayMinutes`, `LateAircraftDelay`, `Reporting_Airline`, …). The targeted generators never emit aviation schemas, so this is a belt-and-braces gate against breadth-bucket leakage.
3. **Numeric-answer hit:** contains any of the problems' ground-truth answers as a token (44.1, 6538, hour 13, 0.7212, …) in a matching context.
4. **Domain overlap for Target C:** the trajectory's dataset is on the held-out-datasets denylist if it is ever promoted into dsbench as a new problem (keep the generator's datasets and dsbench's datasets disjoint).

The gate is a script (`sftgen/decontaminate.py`) run over the rendered mixture before training; it emits a report (rows scanned, rejected, by rule) that is **committed** with the run (ADR-003 reproducibility contract).

## Validation — how we know it worked (and that it's real)

1. **Primary (independent):** a small **held-out targeted probe** — ~8–12 fresh problems exercising the *same three skills* on schemas/domains/dialects the training data never used (e.g. a Postgres weekday-numbering question on the retail schema; a conditional-null denominator on the payments ledger; an ML-delivery task on a held-out dataset). Built in the ADR-003 harness, run k=5 before/after. **This moving is the real claim.**
2. **Secondary:** re-run the dsbench 23-problem k=5 sweep before/after. Expect A/B/C problems to rise; treat any single-problem delta < ~2/5 as noise (k=5 binomial). Because the training data is decontaminated + different-distribution, a dsbench rise that *tracks* the probe rise is a generalisation signal; a dsbench rise *without* a probe rise is an overfitting alarm.
3. **No-regression:** ADR-001 Gate 2 thresholds unchanged — ≤1 pt on IFEval/MMLU-Pro/GPQA subsets, ≤2 pt on LiveCodeBench/HumanEval+, MTP acceptance drop ≤5 pt. The targeted slice is small precisely so it cannot blow these.

## Consequences

- **Easier:** the sandbox that *measures* the model now also *manufactures* its training data, with the same oracle guaranteeing quality — a tight, auditable loop. The generators are reusable for future targets (measure a new gap → add a generator).
- **Harder / watch:** (i) three narrow behaviours risk over-specialisation — mitigated by the 20% cap, breadth, replay, and the regression thresholds; (ii) Target C trajectory generation is the expensive, fiddly part (teacher-in-sandbox, long trajectories, oracle harnessing on new datasets) — build it last and keep the pilot to a handful of datasets; (iii) convention data can become rote (the model learns "say ISO" without understanding) — mitigated by multi-dialect contrast and the thinking traces that *derive* the numbering; (iv) the held-out probe must stay genuinely held out — never fold it back into training.
- **New surface:** a `sftgen/` package (three generators + decontaminate + render) and throwaway Postgres/MySQL/DuckDB containers on `avnet` for cross-dialect verification. Pin them like the sandbox.

## Action items (gated; slot under ADR-001 Gate 1)

1. [ ] Scaffold `projects/dsbench/src/dsbench/sftgen/` (or a sibling package) with the render/decontaminate/provenance plumbing and the corrected chat-template renderer (tool-call XML, assistant-only labels, thinking on/off).
2. [ ] **Target A generator** first (cheapest, teacher-free, highest provenance): synthetic multi-domain schemas, four dialects, execution-verified, ~0.5M tokens; commit the decontamination report.
3. [ ] **Target B generator:** trap-family templates + licence-clean teacher traces, execution-filtered, ~0.6M tokens.
4. [ ] **Target C generator** last: wrap ~4 held-out public datasets in the ADR-003 harness, run a licence-clean teacher agent, keep only oracle-passing trajectories, ~0.9M tokens.
5. [ ] Build the **held-out targeted probe** (~8–12 problems, disjoint from both the generators and dsbench) and baseline the *current* Qwen3.6 on it (so the "before" exists before any training).
6. [ ] Assemble the **Gate 1 pilot 1M mixture** (proportional down-sample), run `decontaminate.py`, and hand it to ADR-001 Gate 1.
7. **Kill / re-plan:** if the held-out probe cannot be moved by the targeted slice even when dsbench moves, the gain is memorisation — stop, and treat the gaps as needing method changes (more/better traces, or RL-style execution feedback) rather than more SFT volume.
