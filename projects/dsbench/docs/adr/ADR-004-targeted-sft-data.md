# ADR-004: Targeted SFT data for the Qwen3.6-35B-A3B fine-tune (the three measured dsbench gaps)

- **Status:** Revision 1 (2026-09-19/20) specified the targeted slice of ADR-001's Gate 1 pilot and Gate 2 run; both were trained. **Revision 2 (2026-09-29, proposed)** specifies the whole mixture for the thinking-on retrain (ADR-001 Gate 2 items 4-5) and is at the end of this document. It supersedes Revision 1's rendering (no empty think blocks), Target A's prompt, reasoning and timezone family, Target C's agent, and the volume table; the targets, provenance, decontamination and validation stand, extended there.
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

> **Revision 2 changes the rendering:** every row is thinking on, with its reasoning. No row trains an empty think block, and the `<think>` opener belongs to the prompt, not the label. Gate 2's trained `<think>` tokens are what its adapter leaked (`reports/gate-evals/20260924-gate2-battery.md`).

### Target A — SQL-dialect date/time conventions  *(generator: `sftgen/dialect_conventions.py`)*

> **Revision 2 changes this target:** the prompt asks for the SQL only (the "numeric result" taught the adapter to invent numbers), the reasoning is hint-conditioned generation by the base (the base's own traces are wrong on these conventions 72% of the time), and the timezone family states its offsets (its truth used fixed summer offsets all year).

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

> **DROPPED from the mixture (2026-09-20) — no measured gap.** The held-out probe (`probe_incident_share`) passed **5/5 even with a tempting wrong-base decoy**, and a trajectory investigation of the one dsbench "denominator failure" (`da_delay_attribution`) showed Qwen3.6's **5-cause denominator was exactly correct** — it only scoped the population differently (`Origin IN hubs OR Dest IN hubs` → 40.9 vs the oracle's `Origin IN hubs` → 44.1). So the "systematic denominator error" was a **prompt-ambiguity artifact**, not a reasoning deficiency. Training it would be teaching-to-the-test on an artifact. The `da_delay_attribution` prompt has been disambiguated (a false negative fixed); the generator below is retained as a tool but is **not** in the targeted mixture. This is exactly the kill-criterion check the probe exists for. The original rationale is kept below for the record.

- **Skill:** identify the **correct population and denominator** for a rate/share/attribution question — especially when a field is populated only under a condition (nulls that mean "not applicable", not "zero"), and when "share of X across categories" means *divide by the summed categories*, not by a grand total or an average-of-averages.
- **Source distribution (NOT aviation):** the same synthetic domains as Target A, but shaped to reproduce the trap families: (i) conditional-null columns (a `refund_reason` populated only for returned orders; a `fault_code` only for failed sensor reads); (ii) share-of-total vs share-of-subtotal; (iii) pooled rate vs mean-of-per-group-rates (the `da_weighted_ontime` trap, generalised); (iv) rate with a cancelled/excluded denominator (the `da_all_flights_avg_delay` trap, generalised).
- **Generation + verification:** template the question + the **execution-verified truth** (SQL and pandas agree). Then — because the *reasoning* is the point — generate a thinking trace with a **licence-clean teacher** that must arrive at the verified number; **drop any trace whose answer ≠ truth** (execution-filtered distillation). The kept trace explicitly discusses *which rows are in scope* and *what the denominator is and why*. This is the one target where a teacher earns its keep (natural denominator-reasoning prose is hard to template well), and it stays provenance-clean because the teacher is licence-clean and the label is ground truth.
- **Implementation notes (2026-09-19).** Two decisions emerged in build: (1) **the data is inlined** in each prompt as a small markdown table — a teacher given only a schema recites `SUM/COUNT` instead of a number, so it can't be execution-filtered; with the data small enough to reason over by hand, the *only* thing separating right from wrong is the denominator/population choice, which isolates the measured skill. (2) **The teacher must be a different family from Qwen.** Qwen3.6 and Ornith both fail `da_delay_attribution` every run, so they are low-yield teachers here; DeepSeek was off-disk, so the hosted teacher is **`Ling-3.0-flash`** (inclusionAI, MIT, a 124B/5.1B-active hybrid reasoning MoE), served on dashi via the proven `toolbox` path with `--reasoning-format deepseek`. Live smoke: **8/8** kept, and it does the pooled sum correctly rather than falling into the mean-of-means trap. Four trap families are implemented: `conditional-null-share`, `pooled-vs-mean-rate`, `share-of-subtotal`, `excluded-denominator`.
- **Format / volume:** ~80% thinking rows (the reasoning is the skill), 20% direct. **~0.6M tokens** (~1,200–1,600 rows) across the four trap families.

### Target C — agentic ML-workflow delivery discipline  *(generator: `sftgen/ml_delivery_trajectories.py`)*

> **Revision 2 changes the agent:** the base generates the trajectories, thinking on (Ling only if the base's yield is too low), and a trajectory must fit 8,192 tokens with its reasoning.

- **Skill:** run the full agentic ML loop in a sandbox and **finish it** — inspect the tables, train a reasonable model, **write predictions for EVERY test row to the exact table name and schema stated in the prompt**, then **stop** (no over-tuning past a good-enough bar). The failure being trained out is "competent model, no delivered artifact."
- **Source distribution (NOT the dsbench aviation tasks):** dsbench-*shaped* but on different public datasets loaded into the sandbox (e.g. UCI/OpenML tabular sets — bike-sharing demand, adult-income, telco-churn, a house-price regression), each wrapped in the ADR-003 harness with an oracle that (a) states the exact output-table contract and (b) grades by a held-out metric with a beat-the-trivial-baseline bar — mirroring `ds_*` **mechanics** without reusing their **content**.
- **Generation + verification:** run a **licence-clean teacher** (DeepSeek/Qwen) as the agent inside the sandbox on these held-out tasks; **keep only trajectories that pass the oracle** (correct-schema table, every row predicted, metric beats baseline). The kept trajectory — tool calls, `run_python`, the `CREATE TABLE … ENGINE=MergeTree ORDER BY …` + `insert_df`, and the terminal "done" — becomes a multi-turn training sample. This is execution-verified by definition (a trajectory only exists as data if it *delivered*), which is precisely the behaviour to reinforce. Include a few "recovered" trajectories (model hits an error, fixes it, still delivers) since ADR-003 notes the model *can* self-recover when it commits.
- **Format / volume:** multi-turn tool-call trajectories, thinking on, assistant-only loss on the assistant/tool-call turns; tool results as `<tool_response>` user turns (B.4). Trajectories are long, so **~0.9M tokens** is only ~150–250 trajectories — generation cost is the teacher inference, hours on the box or a few dollars of API (ADR-001 B.4 cost model).
- **Implementation notes (2026-09-20).** The tasks are **synthetic, non-aviation** (`ml_tasks.py`: a widget-defect classification and a delivery-time regression, each with a real learnable signal), **oracle-gated** by `selftest()` (setup→reference→check with no model — widget AUC 0.923, delivery MAE lift 58.6%, both well past their bars) so teacher time is only spent on solvable tasks. The generator (`ml_delivery_trajectories.py`) reuses the measurement harness's executors and stream reassembly but with a **generic, aviation-free** system prompt and tool schemas so trajectories never carry the benchmark schema. Live smoke on the Ling teacher: it tool-called through the loop, **recovered from a mid-run driver error**, wrote the exact-schema table, and passed (AUC 0.894) — exactly the finish-the-loop behaviour to reinforce. The decontamination gate then earned its keep: it flagged that the task prompts had reused the dsbench `ds_*` deliverable boilerplate (`(id UInt32, … Float64) … one row per test id`), which was reworded to keep Target C disjoint from the eval.

### The targeted slice inside ADR-001's mixtures

> **Superseded for the retrain by Revision 2's mixture**, which is budgeted in rows of thinking-on text rather than in tokens of short answers.

The targeted slice is **~1.4M tokens (~14%)** of the Gate 2 10M-token first run (Target B dropped — see its banner); the rest stays as ADR-001 B.4 breadth + replay. Rationale: the narrow behaviours must not crowd out breadth (or the model over-specialises and regresses — the very thing ADR-001 Gate 2's thresholds guard). Replay stays at 25–30%.

| Bucket | Source | Tokens | % of 10M |
|---|---|---|---|
| **Target A — dialect conventions** | `sftgen/dialect_conventions.py` (own-gen, exec-verified) | 0.5M | 5 |
| ~~Target B — denominator reasoning~~ | dropped 2026-09-20 (no measured gap — held-out probe + `da_delay_attribution` trajectory) | — | — |
| **Target C — ML-delivery trajectories** | `sftgen/ml_delivery_trajectories.py` (teacher-in-sandbox, oracle-passed) | 0.9M | 9 |
| Breadth (DS/DE/SWE/general code) | ADR-001 B.4 buckets (+ the freed 0.6M) | 6.1M | 61 |
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

1. [x] Scaffold `sftgen/` with the render/decontaminate/provenance plumbing and the chat-template renderer (assistant-only labels, thinking on/off). **Done** (`schema.py`/`render.py`/`decontaminate.py`).
2. [x] **Target A generator** (teacher-free, highest provenance): synthetic multi-domain schemas, execution-verified across dialects (DuckDB + ClickHouse live; Postgres/MySQL optional via DSN). **Done** — 6 domains × 4 convention families; scales to the ~0.5M-token target.
3. [x] **Target B generator:** four trap families on inline data, licence-clean teacher trace (Ling-3.0-flash), execution-filtered. **Done — 12/12 live yield, then DROPPED from the mixture (2026-09-20):** the held-out probe (5/5 with a decoy) and a `da_delay_attribution` trajectory investigation (Qwen's 5-cause denominator was correct; only the population scope differed) showed no general denominator gap. Generator retained as a tool, not trained on.
4. [x] **Target C generator:** synthetic ML tasks in the ADR-003 harness (`ml_tasks.py`, oracle-gated), teacher-as-agent, keep only oracle-passing trajectories. **Done** — 4/4 live; trajectory shows train→recover→deliver→finish.
5. [x] Build the **held-out probe** (6 problems, 2 per skill, disjoint from both the generators and dsbench) + its runner (neutral prompt). **Done, oracle-gated 6/6.** Still to do: baseline the *current* Qwen3.6 on it (needs Qwen3.6 hosted) to record the "before".
6. [x] Run the **volume generation** (A ~0.5M teacher-free; C ~0.9M on Ling; B dropped; harden incremental writes for the long runs), assemble the **Gate 1 pilot 1M mixture** (proportional down-sample), run `decontaminate.py`, and hand it to ADR-001 Gate 1. **Done:** the Gate 1 pilot mixture (2026-09-21) and the Gate 2 mixture (2026-09-23, 23,640 rows).
7. **Kill / re-plan:** if the held-out probe cannot be moved by the targeted slice even when dsbench moves, the gain is memorisation — stop, and treat the gaps as needing method changes (more/better traces, or RL-style execution feedback) rather than more SFT volume.

---

# Revision 2 (2026-09-29, proposed): the mixture for the thinking-on retrain

## What changed since Revision 1

- **The Gate-2 adapter failed its acceptance battery** (`reports/gate-evals/20260924-gate2-battery.md`),
  and every cause was in the data:
  - `<think>` was a trained token in 27,690 assistant turns, 25,928 of them the template's empty
    block. The adapter learned to open a block of its own after that one: 54 of 163 HumanEval+
    replies.
  - Trained turns were short (median 94 tokens), and so were the adapter's replies: on MMLU-Pro it
    reasoned half as long as the base (median 386 tokens against 728).
  - All 896 rows that offer tools open with a call, and the adapter called tools that didn't fit
    (BFCL irrelevance 89.2 → 67.1).
  - One Gretel SQL answer in five fails in SQLite against its own schema, and BIRD fell 9.1
    points.

  Held-out loss saw none of it: its forgetting control improved (assistant loss 2.05 → 0.94)
  while IFEval fell 4.6 points.
- **The owner's decisions for the retrain (ADR-001 Gate 2 item 5):** thinking on, in training and
  in serving; LoRA rank 4; a 12,288-token reply cap in the agent harness.
- **Longer steps** (`reports/gate-evals/20260928-seq4096-enablement.md`): the stack trains at 2048,
  4096 or 8192 tokens per step, at 296, 239 and 182 tokens/s. Its kernels are keyed on exact token
  counts, so every step is a full block of one of these lengths.
- **The reasoning pilot** (`reports/gate-evals/20260928-reasoning-pilot.md`) ran the base, thinking
  on, on 200 mixture prompts:
  - a row averages about 3k tokens, against about 420 in Gate 2's mixture;
  - 95% of rows fit 8,192 tokens whole, 85% fit 4,096;
  - the box generates 127 tokens/s with 8 slots;
  - on Target A the base is wrong in one systematic way: 9 of 32 ClickHouse answers verify,
    because it takes `toDayOfWeek` to number Sunday as 1;
  - the timezone family's truth assumed fixed summer offsets all year, which its prompt doesn't
    state.

## Principles

1. **Every row is rendered the way the model is served: thinking on, with its reasoning.** No row
   trains an empty think block. With `<think>` in the prompt, an empty block teaches closing it at
   once, which is skipping the reasoning.
2. **Almost every row is the base's own output.** The base answers every prompt, thinking on. The
   verified pools keep only the answers that pass a check (rejection sampling); replay keeps every
   answer that finishes. The adapter learns which of its own answers to prefer, and nothing of
   another model's style or length, which is where Gate 2's drift and brevity came from. Two
   exceptions, both named below: Target A's hinted traces, and a teacher for Target C if the
   base's yield is too low.
3. **New behaviour comes only from verified pools.** Target A, the SQL pool, Target C and the tool
   rows each pass an execution or schema check.
4. **Nothing is made in the image of a benchmark that only checks for regressions.** ADR-001
   watches IFEval, MMLU-Pro, GPQA, LiveCodeBench and HumanEval+ for losses. Replay built from their
   item types (IFEval's verifiable constraints, say) would hide a regression, not prevent one. The
   base's own replies carry the length, so replay prompts are chosen for breadth.
5. **Budget in rows.** 10M tokens buys about 3,300-4,000 thinking-on rows, not 23,640.

## The mixture (proposed)

10M tokens at 8,192 tokens per step: about 15 hours of training at 182 tokens/s.

| Pool | Tokens | Rows (≈) | Prompts | Kept if |
|---|---:|---:|---|---|
| Target A: SQL-dialect conventions | 0.8M (8%) | 450 | `dialect_conventions.py`, four dialects, asking for the SQL only | its SQL returns the truth on the dialect's sandboxed engine |
| SQL on real schemas | 1.0M (10%) | 500 | SynSQL-2.5M (Apache-2.0), SQLite, each prompt built from its row's database | its result equals the gold SQL's on that database |
| Target C: ML-delivery loops | 1.0M (10%) | 150 | `ml_tasks.py` | the task's oracle passes, and the loop fits 8,192 tokens |
| Tool calls where a tool fits | 0.4M (4%) | 250 | generated tool lists, gold calls and requests; no BFCL item | the call matches the gold call |
| Tool lists that don't fit | 0.3M (3%) | 200 | the same tool lists with unrelated requests | no call |
| Code and SWE | 2.5M (25%) | 800 | opencoder-edu, SWE-Swiss | its tests pass, where the source has them |
| General replay | 4.0M (40%) | 1,600 | clean-origin prompts (below) | the reply finishes |

Every reasoning trace and answer is the base's, thinking on, unless the next section says
otherwise. Every pool also drops replies that hit the generation cap, replies with no reasoning,
and rows over 8,192 tokens.

The verified pools (the first five) take 35% of the tokens; Revision 1's targeted slice took 14%.
The share can be higher because those rows are the base's own answers too, filtered to the right
ones: they pull the model toward its own best behaviour, not toward another model. Replay is 40%,
as ADR-001 Gate 2 item 4 prescribes for the retrain.

**Replay prompts.** The licensing gate in `sftgen/breadth/sources.py` admits only the Tulu 3
subsets of clean origin. The persona sets were written by GPT-4 models, prompts included, and stay
out under ADR-001's taint rule. WildChat's prompts are human-written, but its answers are GPT-4's,
so the gate excludes it whole (decision 5). Proposed:
- oasst1, with history where the conversation has it;
- Aya (human-written, multilingual);
- SciRIFF;
- FLAN v2, in a small share;
- GSM8K train for math (MIT, human-written), to be registered in the gate.

No Robots is on the gate's allowlist, but its licence is CC BY-NC 4.0, so it stays out of
anything redistributable. The allowlist needs that fix.

## Where the reasoning comes from

- **Target A: hint-conditioned generation (STaR's rationalisation).**
  - The *generation* prompt states the convention the row tests, for example that ClickHouse's
    `toDayOfWeek` numbers Monday 1 through Sunday 7.
  - A trace is kept if its SQL verifies and it doesn't cite the hint: no reference to having been
    told, no verbatim copy.
  - The *training* prompt has no hint, so the trace teaches reasoning to the convention, not
    reading it.

  The base's own verified traces join them (28% of the pilot's ClickHouse rows). Two fixes come
  first:
  - The timezone family states each city's UTC offset in the question, as dsbench's
    `da_utc_peak_hour` does. The fixed offsets become the question's premise, so the truth is
    right by construction. What is left to learn is the direction of the conversion, which is
    the measured miss.
  - The Postgres, MySQL and DuckDB answers get sandboxed engines, since model SQL runs only in a
    sandbox. Until then only ClickHouse answers can be verified. **Built 2026-09-29.**
- **SQL.** SynSQL-2.5M was generated with open-source models, per its card. It has 16,583 SQLite
  databases, and its rows are keyed by database. Each prompt is therefore built from its row's
  database in the battery's BIRD format: the DDL and three rows per table, with SQLite named. A row
  is kept if its result equals the gold SQL's. The gold SQL must run first, as the pilot required
  of Target A's truth. BIRD and Spider train (CC BY-SA 4.0) are optional; if used, they are flagged
  for the publish gate's licence audit.
- **Target C.** The base runs the ADR-003 loop on the Target C tasks, thinking on.
  - A loop has one user message, so the template keeps every assistant turn's reasoning, and each
    turn is labelled with it.
  - Gate 2's loops had a median of about 3k tokens with no reasoning. An agentic pilot of about 20
    tasks therefore measures yield and length before any volume run.
  - Loops over 8,192 tokens are dropped. If most are, the tasks shrink (fewer turns): a
    16,384-token step would need new kernel keys, as 4096 did.
- **Tool rows.** A generator writes a tool list and a gold call, then a request the call answers.
  The base's call must match the gold call on function, required arguments and values: the check
  BFCL's AST checker makes. Decline rows pair a tool list with a request no tool serves. The base
  declined 89.2% of BFCL's irrelevance items with thinking off, so these should come easily. No
  BFCL item or schema is used.
- **Replay, code and SWE.** The base's answers, sampled as in the pilot (temperature 0.6, top-p
  0.95, top-k 20), one per prompt.

## Rendering and packing

- **Labels.** With thinking on, the serving prompt ends in `<|im_start|>assistant\n<think>\n`. The
  label starts after it and covers the reasoning, `</think>`, the answer and `<|im_end|>`.
  Gate 2's `tokenize_masked.py` labelled everything after `<|im_start|>assistant\n`, the `<think>`
  opener included. **Built 2026-09-29** (`thinking_record`): the label starts at the end of the
  served prompt, rendered with the generation prompt and required to be a prefix of the row, as
  text and as tokens.
- **Tokenization matches llama.cpp's.** Found while building the labels: transformers' GGUF
  converter left `<think>`, `</think>` and the tool-call tags as BPE pieces (`<think>` = `<th`
  `ink` `>`), which llama.cpp keeps whole. Gate 1 and Gate 2 trained on the split form. The build
  now registers the GGUF's 33 CONTROL and USER_DEFINED tokens (`match_llama_tokenization`). On the
  pilot's 200 prompts, the token counts then equal llama-server's exactly, where they had been 2
  over on every one (`reports/gate-evals/20260929-thinking-rendering-packing.md`).
- **Earlier turns.** The template keeps reasoning only in assistant turns after the last user
  message; earlier assistant turns appear without it. They are context and get no label. Gate 2
  labelled 2,595 of them, and labelled, they train short answers with no reasoning: Gate 2's
  brevity again. A replay prompt with history gets one new assistant turn, the base's; the history
  stays as the dataset wrote it.
- **Packing.** Rows are bin-packed into 8,192-token blocks, never split: first-fit decreasing,
  with the rest of each block padding with no label. `tokenize_masked.py` packed Gate 2 as one
  stream cut every 2,048 tokens. 2,158 assistant turns ran over a block edge, and each one's end
  trained in the next block without its prompt. **Built 2026-09-29:** on the pilot's 199 finished
  rows, 190 packed into 56 blocks, 97.7% full. Rows are separated by `<|endoftext|>`, and neither
  separator nor padding gets a label.
- **Open: rows in a block see each other.** Full attention and the GatedDeltaNet state both carry
  from one row into the next. Checked in the code on 2026-09-29:
  - the model resets the GatedDeltaNet state when given row lengths (`cu_seq_lens_q`);
  - its 4-tap convolution's fallback ignores boundaries;
  - attention's variable-length path is unverified for the aiter kernel;
  - the recipe's collator passes no lengths.

  A GPU check (a block with row lengths against its rows run alone) and a collator patch are next.

## Budget and time (estimated from the pilot's throughput)

- **Generation**, with production stopped:
  - replay, code and SWE: about 2,500 prompts (5% of rows run over 8,192 tokens and are dropped)
    at 2,560 tokens a reply: about 6.4M tokens, 14 hours;
  - the verified pools, about 1,400 kept rows: at a 30-60% yield, 3-7M generated tokens,
    7-15 hours;
  - Target C: sandbox time on top;
  - in all, about 25-40 hours of box time.
- **Training:** 10M tokens at 8,192 ≈ 15 hours, plus a few percent of padding.

## Validation

- **A thinking-on mini-battery on checkpoints** (ADR-001 Gate 2 item 4):
  - IFEval, BFCL irrelevance, BIRD's SQLite error count and HumanEval+, as the battery report
    proposed;
  - plus the median reasoning length against the base's, the brevity check;
  - with thinking on there is no empty block to leak after, so the think-leak check becomes the
    share of replies that never close their reasoning.

  The battery report sized it at an hour with thinking off. Reasoning makes each item several
  times longer, so it runs on subsets of a few hundred items, which is where each Gate-2 failure
  already stood far past its noise. Held-out loss is not a gate; Gate 2 showed why.
- **The full battery, re-baselined with thinking on for both states.**
  `20260924-gate2-battery.md` measured the base with thinking off, and those numbers don't carry
  over. Token limits must be sized for reasoning; GPQA's 4,096 bound it in Gate 2.
- **The held-out targeted probe and dsbench (k = 5), as in Revision 1.** The owner's suite runs
  through pi with its 12,288-token reply cap, recorded in each run.
- **Decontamination before training, against every battery benchmark.** `decontaminate.py` checks
  dsbench only, and Gate 2's exposure to the battery was measured after training
  (`battery/contamination.py`). The retrain's gate also rejects rows on the battery's 13-gram
  index. SWE rows still drop SWE-bench-Verified's repositories. **Built 2026-09-29**
  (`decontaminate.py --battery-items`, `reports/gate-evals/20260929-battery-decontamination.md`):
  - a row is rejected when it reproduces a fifth of a battery item's 13-grams, or a short item
    whole;
  - on Gate 2's mixture it touches the same 44 items as the after-the-fact check and rejects 2
    rows.

## Decisions for the owner

1. **Budget:** 10M tokens (about 15 hours of training and 25-40 of generation), or more.
2. **Target A's reasoning:** the hint-conditioned base (proposed: on-policy and licence-clean), or
   a teacher (Ling).
3. **SQL:** SynSQL-2.5M alone (proposed), or also BIRD and Spider train under CC BY-SA 4.0.
4. **jupyter-agent (559 rows in Gate 2) and DataMind (640):** their answers are other pipelines'
   text, which principle 2 excludes. Drop them (proposed: Target C, the SQL pool and the tool rows
   carry the DS and tool behaviour), or regenerate them with the base as the agent, which needs
   their data in the sandbox.
5. **Replay prompts:** clean-origin sets only (proposed), or also WildChat's first user turns,
   which are human-written; their GPT-4 answers would be replaced.
6. **The proportions above.**

## Action items (Revision 2)

1. [ ] Rendering and packing. **Done 2026-09-29 except the last point**
   (`reports/gate-evals/20260929-thinking-rendering-packing.md`):
   - [x] the `<think>\n` opener stays in the prompt;
   - [x] no labels before the last user message;
   - [x] bin-packing into 8,192-token blocks without splitting rows (`build_masked_dataset.py` now
     defaults to 8192);
   - [x] tests on rendered examples, with the model's own chat template;
   - [x] found on the way: the tokenizer now splits text as llama.cpp does;
   - [ ] rows in a block see each other: a GPU check and a collator patch that passes row lengths.
2. [ ] Target A:
   - [x] offsets stated in the timezone family. **Done 2026-09-29:**
     - the question states each city's offset, daylight or standard time, drawn per instance,
       and holds it for every row;
     - the truth follows from the stated offsets by construction;
     - 36 of 36 generated rows verify on ClickHouse and DuckDB, and all 36 pass both
       decontamination gates;
     - the synthetic tables are unchanged, so old rows still rebuild.
   - [x] the SQL-only prompt. **Done 2026-09-29:** the system prompt and the questions ask for
     the query only, and a generated row's answer is the SQL. The verified number stays in the
     row's verification record.
   - [x] sandboxed Postgres, MySQL and DuckDB engines. **Done 2026-09-29**
     (`sandbox/docker-compose.yml`, `engines.available_engines(sandboxed=True)`):
     - model SQL runs through a read-only login in Postgres 16 and MySQL 8.4, and in DuckDB 1.5.5
       in a container with no network;
     - the pilot's answers in those dialects verify 10/16, 11/16 and 10/16, against ClickHouse's
       9/32 (`reports/gate-evals/20260928-reasoning-pilot.md`, addendum).
   - [ ] hint-conditioned generation with its hint-citation filter, piloted on about 50 rows per
     dialect. The addendum's numbers say where the base is wrong: ClickHouse weekdays and
     weekends first, then DuckDB weekends. The timezone family goes too, once it is re-measured
     with its stated offsets.
3. [ ] SQL: acquire SynSQL-2.5M's databases, build the prompts from them, write the
   execution-match verifier.
4. [ ] Tool rows: generators for tool lists, gold calls and requests (fitting and not), and the
   gold-call checker.
5. [ ] Target C: the agentic pilot (about 20 tasks, thinking on) for yield and length, then volume.
6. [ ] Replay, code and SWE: drop No Robots from the Tulu allowlist, register GSM8K, select the
   prompts, generate.
7. [x] Decontamination: extend `decontaminate.py` with the battery's 13-gram index. **Done
   2026-09-29**, with short items matched whole and BFCL's schemas indexed
   (`reports/gate-evals/20260929-battery-decontamination.md`).
8. [ ] The thinking-on mini-battery, and the full battery re-baselined for the base.
9. [ ] Assemble, train (rank 4, 8,192 tokens), and gate checkpoints on the mini-battery.
