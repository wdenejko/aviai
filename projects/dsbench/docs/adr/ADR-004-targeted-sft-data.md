# ADR-004: Targeted SFT data for the Qwen3.6-35B-A3B fine-tune (the three measured dsbench gaps)

- **Status:** Revision 1 (2026-09-19/20) specified the targeted slice of ADR-001's Gate 1 pilot and Gate 2 run; both were trained. **Revision 2 (2026-09-29; its decisions taken 2026-10-03 and 2026-10-05; trained and gated 2026-10-06; tested against the base 2026-10-06/07; its recall round generated 2026-10-07; Revision 2.1 with the round's rows trained and gated 2026-10-07/08, tests running)** specifies the whole mixture for the thinking-on retrain (ADR-001 Gate 2 items 4-5) and is at the end of this document. It supersedes Revision 1's rendering (no empty think blocks), Target A's prompt, reasoning and timezone family, Target C's agent, and the volume table; the targets, provenance, decontamination and validation stand, extended there.
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

# Revision 2 (2026-09-29; decisions 1-8 taken 2026-10-03 and 2026-10-05; trained and gated 2026-10-06): the mixture for the thinking-on retrain

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

10M tokens at 8,192 tokens per step: about 15 hours of training at 182 tokens/s, or about 9 at
the packed rows' measured step ("Budget and time").

| Pool | Tokens | Rows (≈) | Prompts | Kept if |
|---|---:|---:|---|---|
| Target A: SQL-dialect conventions | 0.8M (8%) | 450 | `dialect_conventions.py`, four dialects, asking for the SQL only | its SQL returns the truth on the dialect's sandboxed engine |
| SQL on real schemas | 1.0M (10%) | 500 | SynSQL-2.5M (Apache-2.0), SQLite, each prompt built from its row's database | its result equals the gold SQL's on that database and on three bigger variants of it |
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
- Aya (human-written, multilingual). **Acquired 2026-09-30** (`tulu3_aya`): 99,964 of the
  subset's 100,000 rows in 71 languages (34 are over the length limit, and dsbench's gate drops 2).
  Two thirds were written from scratch; a third are human edits of machine-generated text. Each
  row keeps its language and annotation type, so selection can balance them;
- SciRIFF. **Acquired 2026-09-30** (`tulu3_sciriff`): 4,902 of the subset's 10,000 rows. SciRIFF
  repurposes existing scientific-literature datasets as tasks, and its card lists each source's
  licence:
  - kept: the 24 tasks under CC BY, CC0, Apache-2.0 or MIT, less 2 rows dsbench's gate drops;
  - out: one task under CC BY-NC (271 rows), one under GPL-3.0 (1 row), and 19 tasks with no
    licence listed (4,824 rows; decision 5);
- FLAN v2, in a small share;
- GSM8K train for math (MIT, human-written). **Registered 2026-09-30** (`gsm8k` in the gate):
  - hired writers wrote the problems and their worked solutions. OpenAI released them, but no
    model wrote them, so ADR-001's teacher rule doesn't apply;
  - the train split only, since test is GSM8K's benchmark;
  - all 7,473 rows pass: dsbench's gate drops none, and none shares a 13-gram with a battery item;
  - each row keeps its gold number in `meta`, so the base's answers can be checked (decision 5).

No Robots is on the gate's allowlist, but its licence is CC BY-NC 4.0, so it stays out of
anything redistributable. The allowlist needs that fix. **Fixed 2026-09-29:**
- No Robots is out of the allowlist, and so is a dead entry that matched no subset;
- the allowlist now names exact subsets with their licences (`sftgen/breadth/sources.py`), so a
  renamed or new subset fails closed;
- no No Robots row was ever acquired: the Tulu pool so far is oasst1 and FLAN v2 only.

The mixture's card gives FLAN v2 no licence (decision 5).

Aya and SciRIFF come from their subsets' own repos, at pinned revisions. The rows are the
mixture's, message for message (checked 2026-09-30), but the mixture keeps only the messages,
and the SciRIFF gate needs each row's task. Both pools pass the battery gate with no row rejected.
Below the line:
- SciRIFF has 3 rows, sharing digit runs, at most 3.2% of an item's 13-grams;
- Aya has 69 rows. 68 share digit runs, at most 8%. One English row shares 15.5% of IFEval item
  2859's 13-grams, a passage both contain.

Selection can leave out every row with an overlap. The pools are git-ignored;
`data/sft/rev2_breadth_manifest.json` records their revisions, hashes and gate results.

**Prompt selection (built 2026-09-30, `sftgen/select_prompts.py`).** A prompt is a row's messages
up to its last user turn, history included, and its items run through `reasoning_pilot.py
generate` as they are. A row is eligible when its prompt:
- fits in 3,072 tokens (chars/3.5), leaving 5,120 for the reply. That covers the pilot's p90
  reply in every pool but opencoder's (10,195), whose prompts are short: no prompt cap helps
  there;
- shares nothing with the battery, below rule 4's line included;
- carries its check where the source has one: GSM8K's gold number, and opencoder's tests;
- isn't a duplicate.

Each pool's quota is its share of the bucket's kept rows, divided by the share it is expected to
keep after generation: 0.95, and 0.75 for opencoder. It lost 6 of 32 rows to length in the
pilot, and its rows must also pass their tests, as the base's closed HumanEval+ replies did
94.5-95.9% of the time in the mini-battery's calibration; 0.8, until 2026-10-03, counted length
alone. Aya is sampled evenly across its 71 languages and SciRIFF across its 24 tasks. OpenCoder
was acquired again, because the Gate-2 pool kept only a flag where the tests should be. GSM8K's
and OpenCoder's revisions are pinned now, like Aya's and SciRIFF's.

SWE-Swiss was acquired whole, at a pinned revision, because the first selection fell 11 short on
Gate 2's pool, which stopped at its token cap after 293 rows. Its prompts carry repository code:
the median is about 8,900 tokens. Of its 10,254 rows:
- 64 name a SWE-bench-Verified repository and are dropped (ADR-001);
- 7,922 are over ADR-001's 8,000-token limit, counting DeepSeek-R1's answer, which Revision 2
  discards. The limit may also lean the pool toward prompts with shorter replies;
- 2,267 are kept, and 1,609 of their prompts are eligible.

The selection (seed 20260930), 2,696 prompts:

| Pool | Share | Selected | Eligible | Dropped |
|---|---:|---:|---:|---|
| oasst1 | 0.40 | 674 | 7,118 | 7 duplicates, 4 long, 2 battery |
| Aya | 0.20 | 337 | 90,437 | 9,514 duplicates, 8 battery, 5 long |
| SciRIFF | 0.20 | 337 | 4,223 | 670 long, 9 duplicates |
| GSM8K | 0.20 | 337 | 7,473 | none |
| FLAN v2 | 0 | 0 | 2,361 | (decision 5) |
| opencoder | 0.75 | 800 | 8,142 | 1,719 duplicates, 100 battery, 39 tests (below) |
| SWE-Swiss | 0.25 | 211 | 1,609 | 639 long, 11 duplicates, 8 battery |

- 249 of the oasst1 prompts carry history.
- Expected after generation: about 1,600 replay rows and 800 code rows.
- **OpenCoder's prompts now show a test (2026-10-02).** The tests call the function by the name
  the dataset's own answer gave it, and expect that answer's return format, but only 19 of the
  750 prompts first selected named the function. A right answer under another name would have
  failed every test. Each prompt now ends with its first test, as MBPP's prompts show theirs
  ("Your code should pass this test, and others like it"), and the check runs all of them.
  - Duplicates are still found by the instruction alone.
  - The battery gate sees the shown test, and drops 56 more rows. Every test it catches holds a
    literal list of small numbers (a run of numbers, a 0/1 grid, the permutations of [1, 2, 3],
    a small matrix), shared with DS-1000, LCB, HumanEval+ or BFCL items, at most 16% of one.
  - 39 rows are dropped whose tests span several lines (building a tree): the dataset stores
    those lines out of order.
  - The new candidates change the sample: 58 of the 750 instructions are the same. Nothing had
    been generated. The other pools' 1,896 prompts are unchanged, ids included.
  - The dataset's own answers pass their tests on 748 of the 750 items, run through the same
    extraction and sandbox as the base's replies (`replay_verify.py --gold`, in docker on the
    Mac). One answer needs sympy, which the sandbox lacks; the other returns a set's order. Both
    items stay: the base's reply is judged on its own.
- **OpenCoder's quota rose from 750 to 800 (2026-10-03)**, with its keep rate. The first 750
  picks are the same, and so are the other pools' prompts. On the box's own sandbox, where the
  check runs, the dataset's answers pass the tests of all 800 items: that image has sympy, and
  the docker stand-in on the Mac didn't.
- Each pool draws from its own seeded generator: SWE-Swiss's new pool left the other 2,435
  prompts unchanged.
- `data/sft/rev2_prompts_manifest.json` records the parameters, each pool's counts, languages
  and tasks, and the hashes of the pools and the output.

## Where the reasoning comes from

- **Target A: hint-conditioned generation (STaR's rationalisation).** Its pilot ruled it out, and
  a reasoning prefill is proposed in its place (decision 2). Both are described below, in order.
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

  **The generator, built 2026-09-30** (`sftgen/target_a_hints.py`):
  - The hint is one sentence:
    - for a weekday or weekend row, the dialect's weekday numbering, not the weekend's numbers;
    - for a month row, the month function;
    - for a timezone row, the direction rule.

    Every weekday and month hint is checked on the four sandboxed engines before any item is
    built, each function on all seven days. Tests tie each hint to the generator's own numbering.
  - A reply is kept when it finished with reasoning, is one ```sql block and nothing else (the
    SQL pool's rule), returns the truth on the sandboxed engine, and doesn't read as told. It
    reads as told when it:
    - names a hint;
    - talks about the framing;
    - attributes the hint's terms to the prompt ("the prompt explicitly says toDayOfWeek ...");
    - copies 8 tokens of the hint's wording.

  **Piloted 2026-09-30** (`reports/gate-evals/20260930-target-a-hints-pilot.md`): 192 rows, 12
  per dialect and family, each answered plain and hinted, in one GPU window.
  - **The gap is ClickHouse's weekday numbering.**
    - Plain, 1 of 12 ClickHouse weekday rows verify, and 0 of 8 weekend rows whose question is
      unambiguous. All 24 of those traces consider Sunday = 1.
    - Month buckets verify 48 of 48.
    - The timezone family, with its stated offsets, verifies 44 of 48.
    - Postgres and MySQL answer their weekdays right.
  - **The hint fixes the answers, but the base reads it back.**
    - With the hint, every weekday row verifies, and so does every weekend row whose question is
      unambiguous.
    - Stated plainly, the hint left 11 of 96 weekday and weekend traces clean. Framed as the
      base's own knowledge, it left 4, because the base quoted the framing.
    - The first framing's clean traces run less than half the base's usual length.

    **Hint-conditioned generation can't supply Target A as designed.**
  - **Next**, each about half an hour of box time: prefill the base's own reasoning with the
    convention, so no prompt text states it, or sample plain ClickHouse prompts at scale (the
    base verified 1 of 12, reasoning right). A teacher stays the fallback (decision 2).
  - **Found on the way:**
    - One weekend phrasing, ", by order_ts?", was answered with GROUP BY 30 times in 32. It made
      most of the weekend misses outside ClickHouse, and its wording is fixed.
    - The citation filter was calibrated on the pilot's traces, and flags none of the 272 that
      never saw a hint.

  **The reasoning prefill, built 2026-10-01** (`sftgen/prefill.py`; items from
  `target_a_hints.py prefill-items`):
  - The convention is written into the base's thinking block, as if the base had recalled it,
    and the base continues from there. The training row is the plain prompt and the whole trace,
    prefill included, so the row holds nothing for the trace to cite.
  - Two placements, piloted side by side:
    - `start` opens the thinking block with the convention;
    - `recall` lets the base answer plain first, cuts its trace at the sentence where it first
      turns to the weekday function or states a numbering, and writes the convention there. The
      row keeps the base's own opening, and the one sentence the base didn't write is the one it
      gets wrong.

    On the hint pilot's 24 plain ClickHouse traces, every cut falls before the base names the
    function or a numbering, a median of 469 characters in (8% of the trace), and never inside a
    drafted query.
  - The sentence is written in the base's own register. It names the alias `dayOfWeek`, which
    the base names before `toDayOfWeek` in 14 of those 24 traces. It also says that Sunday = 1
    is MySQL's numbering, the belief's likely source. Each function it names is checked on its
    engine.
  - A prefilled reply is checked like a hinted one, on what the base wrote after the prefill. It
    may refer back to the prefill ("as noted above"), since the row keeps it. It may not credit
    the prompt with the convention, or name a hint.
  - A prefill can't go through the chat endpoint: the template closes an assistant message's
    thinking block. So the server renders the prompt with its own template (`/apply-template`),
    and `/completion` continues the prefill. Checked locally against the chat endpoint: with
    nothing prefilled, both paths return the same reply, token for token.
  - **The pilot:**
    - 24 ClickHouse rows, 12 weekday and 12 weekend;
    - each answered 4 times plain, 4 times `start`, and up to 4 times `recall`, cut from the
      plain replies: at most 288 replies, about an hour of box time;
    - the plain replies also measure the other route: how often the base gets a row right alone,
      sampled several times.

  **Piloted 2026-10-01** (`reports/gate-evals/20261001-target-a-prefill-pilot.md`), in one
  66-minute window:
  - **`recall` supplies the rows.** 95 of 96 replies verify, none cites anything, and every row
    has kept replies. The traces keep the base's length: a median of 1,541 and 1,561 reasoning
    tokens, against 1,504 and 1,446 for its plain replies. The one miss is an arithmetic slip
    under the right numbering.
  - **`start`** keeps 92 of 96, but its traces run two-thirds of the base's length.
  - **Plain sampling can't.** 7 of 96 verify, and 18 of the 24 rows were never right in 4 tries;
    250 rows would take about 19 hours of box time.
  - **The old belief comes back as a doubt.** About half of the kept prefilled traces raise
    Sunday = 1 again before settling on ISO ("is there any chance `toDayOfWeek` returns 1 for
    Sunday"). Keeping only the `recall` traces that don't still leaves a kept reply on 23 of 24
    rows, at the base's length.
  - **For scale:**
    - 250 rows take about 1 hour of box time keeping the doubting traces, 2 without;
    - paraphrase the sentence;
    - stop the plain phase at 512 tokens, since every cut fell within about 200.

  **The volume, built 2026-10-02** (`target_a_hints.py volume-items`, as decision 2 proposes;
  `data/sft/rev2_target_a_manifest.json`):
  - **ClickHouse's weekday and weekend rows go through `recall`.**
    - The sentence comes in 4 wordings, drawn per row. Each makes the pilot sentence's claims,
      filled in from the checked functions: the function, its alias, its numbering, that the
      numbering is ISO, and that Sunday = 1 is MySQL's.
    - The plain phase stops at 512 tokens (`max_tokens` on the item). The window's `splice`
      step cuts each reply and writes the convention there, and the base continues.
  - **The rows follow the prompts.** The model sees only the question, never the table, so a
    family has only as many prompts as questions times domains: 126 weekday prompts but 18
    weekend ones. Equal rows a family would have repeated each weekend prompt about 16 times. So
    weekdays get 450 rows and weekends 72 (124 and 18 prompts drawn). That is about 2 kept traces
    a prompt in both, once the doubting traces are dropped.
  - **The other 14 cells** come from the base's plain replies, 18 rows a cell (227 distinct
    prompts), kept where they verify.
  - **The doubt filter.** It is now `target_a_hints.doubts`: a sentence after the prefill that
    states Sunday = 1 and doesn't name MySQL, counted only in ClickHouse's weekday and weekend
    rows.
    - `verify` records the count. The assembler drops those rows by default, as proposed;
      `--target-a-doubts keep` keeps them.
    - On the pilot's kept replies it reproduces the report's counts exactly: 49 of 95 `recall`,
      40 of 92 `start`, 7 of 7 plain.
  - **The items.** 522 plain-phase items and 252 cell items. Every hint check passed on the four
    sandboxed engines, and neither the generator nor the battery gate rejected a row.
  - **About 3.5 hours of box time**, by the pilot's lengths: 0.3M tokens for the plain phase,
    0.8M for recall and 0.4M for the cells.
- **SQL.** SynSQL-2.5M was generated with open-source models, per its card. It has 16,583 SQLite
  databases, and its rows are keyed by database. Each prompt is therefore built from its row's
  database in the battery's BIRD format: the DDL and three rows per table, with SQLite named. A row
  is kept if its result equals the gold SQL's. The gold SQL must run first, as the pilot required
  of Target A's truth. BIRD and Spider train (CC BY-SA 4.0) are optional; if used, they are flagged
  for the publish gate's licence audit. **Built 2026-09-30** (`sftgen/synsql.py`), SynSQL alone,
  as decision 3 proposes:
  - The questions sit in one 9.4 GB JSON array, grouped by database. The build reads 64 KB at
    seeded offsets of the pinned revision and takes the first whole question after each offset,
    one question per database. A question's chance therefore follows the length of the question
    before it. On 150 chunks (2,406 questions), the picks matched the dataset's complexity mix
    within noise.
  - **SynSQL's databases are nearly empty.** In the pool's 1,112 databases, a table holds 2 rows
    at the median and 10 at most. The prompt's example rows are therefore the whole database. On
    the database alone, the gold SQL returned no rows for half the questions read, and a wrong
    query often returns the gold's rows. On 300 items of a first build, all with gold rows,
    mutated gold queries still matched the gold:
    - with an aggregate swapped (AVG for SUM, MIN for MAX), 60% of the time;
    - with the sort flipped before a LIMIT, 89%;
    - with a comparison flipped, 9%.
  - So every query also runs on three variants of its database, and a reply must match the gold
    on all four. This is test-suite accuracy (Zhong, Yu & Klein, 2020).
    - A variant keeps the tables and the rows, and adds new rows up to 100 per table.
    - A new value repeats one the column holds half the time, so the question's literals still
      match rows.
    - A foreign key points at a few parents most of the time. Groups then hold several rows,
      where AVG and SUM differ, and some parents have none, where INNER and LEFT JOIN differ.
    - With the variants, on the pool's 1,112 items, the three mutations matched 15%, 19% and
      0.9% of the time. The aggregate survivors inspected are equivalent to the gold: an
      aggregate the query doesn't return, or a group of one row by construction.
  - Every query runs in a new `sqlite` sandbox container, the gold SQL included, since a model
    wrote it too. The variants are built there as well, because building one runs the dataset's
    CREATE TABLE statements.
    - The container has no network, a read-only root and no capabilities.
    - The runner opens each database read-only, refuses ATTACH and interrupts a query after 30
      seconds.
  - A question is kept when:
    - its style isn't "Multi-turn Dialogue", a conversation pasted into one question (10% of
      those read);
    - its prompt fits in 4,096 tokens;
    - its gold SQL runs on all four databases and returns at most 1,000 rows on each;
    - the gold returns rows on at least one database, and not only NULL, 0 or ''. A result
      that is empty or blank everywhere passes any query that matches nothing, so it checks
      nothing.
  - Of the 1,528 questions whose gold SQL ran, 1,112 were kept. The gold never failed on the
    database itself. It returned no rows anywhere for 341 (22%), only blanks for 37, over 1,000
    rows on a variant for 36, and failed on a variant for 2. Moderate questions lose the most to
    empty results: 28% of those read.
  - The gate rejected 6 questions. Five shared a run of 0s and 1s with a DS-1000 item: flag
    columns in the example rows, 1 or 2 of the item's 150-odd 13-grams. The sixth contained a
    dsbench answer (44.1). None touched BIRD.
  - The prompt has three phrasings. Each names SQLite and asks for the query only, in a ```sql
    block.
  - A reply passes when:
    - it is that one block and nothing else;
    - its rows equal the gold SQL's as a set on all four databases (BIRD's execution accuracy,
      four times);
    - it finished, with reasoning.

    Target A's check takes the last block and allows prose around it. Its generation can adopt
    this rule.
  - 1,112 prompts (seed 20260930), for 500 rows at an expected 0.45, a guess until the base's
    replies are checked (`data/sft/rev2_sql_prompts_manifest.json`):
    - Complex 408, Highly Complex 321, Moderate 282, Simple 101, close to the mix read;
    - prompts of 2,383 tokens at the median and 3,393 at p90;
    - for 409 prompts the gold returns no rows on the database itself, and only the variants
      check them. On a variant the gold returns 8 rows at the median;
    - the build is deterministic: a 30-item run reproduced the pool's first 30 items byte for
      byte;
    - `decontaminate.py --battery-items` passes all 1,112, with no overlap below the line.
- **Target C.** The base runs the ADR-003 loop on the Target C tasks, thinking on.
  - A loop has one user message, so the template keeps every assistant turn's reasoning, and each
    turn is labelled with it.
  - Gate 2's loops had a median of about 3k tokens with no reasoning. An agentic pilot of about 20
    tasks therefore measures yield and length before any volume run.
  - Loops over 8,192 tokens are dropped. If most are, the tasks shrink (fewer turns): a
    16,384-token step would need new kernel keys, as 4096 did.

  **The pilot, built 2026-10-01** (`sftgen/ml_delivery_trajectories.py --thinking`;
  `patches/target_c_pilot_{window,arm,mac}.sh`; `patches/target-c-pilot/`):
  - **The agent is the base.** Thinking on, with Qwen's sampling. Every assistant turn keeps its
    reasoning, and each request resends the earlier turns' reasoning, as the training row will
    show it. A turn cut by the token limit ends the run, never kept.
  - **21 runs:** the 7 task families x 3 datasets (run indices 101-103, past Gate 2's 0-62).
    Before any GPU time, the oracle passed all 21 datasets without a model (`--oracle-only`); at
    index 100 it failed `mlc_churn_rare`'s own reference (AP 0.149 against a bar of 0.15), so the
    pilot starts at 101.
  - **Parallel:** 8 loops at once, one per server slot. Each run has its own ClickHouse database
    and its own working directory in the workspace container. Without one, an agent's relative
    files landed in the repo, which the container mounts as its working directory.
  - **Where it runs:** the box only serves the base; the loop runs on the Mac beside the sandbox,
    through an SSH tunnel. The window holds the server until the Mac releases it, as the battery's
    dsbench phase does.
  - **What it measures** (`measure_trajectories.py`, on the box): each trajectory through
    `thinking_record`, the code that builds the retrain's blocks. That gives its tokens, whether
    it fits 8,192, and whether a turn has empty reasoning, which rejects the row. Also measured:
    yield per family, turns, and failure modes.
  - **Gate 2's trajectories don't carry over.** Run through the same measurement, a sample of 20
    Ling trajectories is all rejected: no turn has reasoning. Even without it, 2 of the 20 are
    over 8,192 tokens (median 2,820).

  **Piloted 2026-10-01** (`reports/gate-evals/20261001-target-c-agentic-pilot.md`), in an
  18.8-minute window:
  - **No teacher is needed.** 20 of 21 runs pass the oracle, and every family passes at least 2 of
    3. All 215 assistant turns carry reasoning, and `thinking_record` rejects no trajectory.
  - **The 8,192-token block keeps 14 of the 20.** A passing loop runs a median of 5,715 tokens.
    The 6 over the block come from the hard families: credit_leak and upsell_join keep 1 of 3
    each, and energy_load 1 of its 2 passes.
  - **The length is mostly code.** Tool-call arguments are 56% of a passing loop's characters,
    tool output 25%, reasoning 18%. `run_python` runs each call in a new process, so every fix
    resends the whole script.
  - **The block selects.** It drops the loops with more failed calls (2.67 erroring turns against
    1.29 kept). It also drops both credit_leak loops that reason through the leak; the kept one
    left the column out without a word.
  - **About 28% of a passing loop's labelled text sits in turns whose call failed.** 16 of 20
    loops have such a turn. The commonest error is reading a ClickHouse result into pandas: 16 of 41 errors.
  - **The miss** is energy_load's trap: lags filled forward across the horizon, validated one step
    ahead. The base's reasoning named the problem and kept the model for its validation number.
  - **The row matches the loop.** It is at most one token longer than the server's count of the
    loop's last request, so generation can apply the 8,192 cap itself.

  Proposed: volume at 8,192 tokens with a quota per family, about 300 runs and 3.5 hours for 150
  rows. A 16,384-token step would keep 19 of the 20 passes, but needs new kernel keys. Whether
  failed turns train is decision 7.

  **Quota mode, built 2026-10-01** (`ml_delivery_trajectories.py --quota`, `generate_quota`):
  - **Each family runs until it has its rows.** Run indices count up from `--run-offset` per
    family. Each dataset passes its own oracle before the agent sees it, so `--oracle-only` is no
    longer a separate step.
  - **A row is kept** when:
    - the oracle passes;
    - every turn carries reasoning;
    - no tool call names the withheld key or the admin's login;
    - the agent ended the loop itself, with `finish` or a final reply (added after the volume
      run, below);
    - the server's count fits the block, with any tool output after the last request counted
      at a token a character (also added after the volume run), and the separator that follows
      the row in its block (added 2026-10-02; no kept row was within a token of the edge).

    The other runs go to `--fail-out` with the reason. Rows over the block keep `oracle_passed`,
    for a longer step later.
  - **A free slot goes to the family furthest from its quota.** Each of its running loops counts
    at its keep rate so far. Simulated on the pilot's keep rates and run times, with 22 rows a
    family over 5 trials:
    - 286 to 336 runs;
    - 3.4 to 4.1 hours, with 7.4 of 8 slots busy;
    - 1 to 4 rows over the 154.
  - **Stopping and resuming.** `--max-minutes` stops new runs before the window ends. `--resume`
    continues from the output files and never reuses a run index. Consecutive model errors stop
    the run.
  - **Found: the answer keys were visible to the agent.** Each task writes its withheld labels
    into the run's own database, where the oracle reads them.
    - Gate 2's Ling loops listed them in 15 of 350 runs and read none. The base never listed
      tables in the pilot.
    - A run that read a key would pass by copying the labels, so its row is now never kept, in
      either mode.
    - **Fixed the same day:** the agent runs as its own ClickHouse login, as in dsbench and the
      probe (ADR-003 §7). The key is refused and drops out of SHOW TABLES; the login has no
      shared data. Checked live: a scripted agent's read of the key was refused, its deliverable
      still graded, and nothing was left behind. `ml_tasks`' oracle gate checks every task's
      login.
  - **For the window,** the default 4-hour hold is tight. Either one window with a 5-hour hold
    and `--max-minutes 270`, or two windows with `--resume`.
  - **The window's scripts, built 2026-10-01** (`patches/target_c_volume_*.sh`, described in
    `patches/README.md`). They take either way:
    - the hold lasts 5 hours by default, and the Mac reads its deadline: no run starts in its
      last 30 minutes;
    - the hold ends early if the Mac never starts, stops sending requests, or the server dies;
    - each window holds in its own directory, and every pass runs with `--resume`, so a second
      window continues the first;
    - the tunnel restarts itself, the generator waits out a lost connection (3 minutes), and a
      run of errors leads to another pass once the server and the sandbox answer.

    Rehearsed against a stand-in on the box. The rehearsal found the box's clock about 2 hours
    behind (NTP off); the Mac converts the deadline into its own clock.
  - **The volume run, 2026-10-01/02** (`reports/gate-evals/20261001-target-c-volume-run.md`): one
    window, 4 hours 40 minutes of generation.
    - 154 rows: 22 a family, but upsell_join 21 (the time limit) and energy_load 23. They hold
      0.80M tokens, 0.38M labelled, against the table's 1.0M.
    - The base passed 275 of 282 runs. Every miss was energy_load's, and there were no errors.
    - 121 passing loops ran over 8,192 tokens, and 73% of the slot time went into runs that
      weren't kept. At 16,384 tokens, 247 of the 275 would train.
    - The agent's login held: no run reached for a key, and no call was refused.
    - Found: 2 runs used up their steps after writing a passing table, and 3 rows end on tool
      output that no request carried. Neither kind was kept here. The selection now drops the
      first and counts the second.
- **Tool rows.** A generator writes a tool list and a gold call, then a request the call answers.
  The base's call must match the gold call on function, required arguments and values: the check
  BFCL's AST checker makes. Decline rows pair a tool list with a request no tool serves. The base
  declined 89.2% of BFCL's irrelevance items with thinking off, so these should come easily. No
  BFCL item or schema is used. **Built 2026-09-30** (`sftgen/tool_rows.py`):
  - 30 tools in 20 domains, written for this generator. A request states every value the call
    needs, in everyday words ("7:30 pm", "euros"), while the schema asks for a format ("HH:MM",
    ISO 4217). So a fit row also checks that the base maps one to the other. Values with two
    common spellings are accepted both ways (a city with or without its country).
  - Decline rows offer the fit rows' tool lists. About 60% of the requests ask for another
    domain's action, and about 40% need no tool at all. A web-search tool could answer anything, so it is never
    offered in a decline row. A general request close to a domain (planning a workout) is never
    paired with that domain's tools.
  - The check applies the rules of BFCL's AST checker. The function must match; every required
    argument must be there, and no argument the schema lacks. Types must hold, though an integer
    passes for a number. Strings are compared the way BFCL compares them, and lists as sets. A
    value the request states must be passed; an unstated optional one may be left out or given
    its default. A row also needs a finished reply with reasoning.
  - The gate rejects an item that shares any 13-gram with the battery, not only a fifth of one.
    The first run lost 56 candidates to BFCL schemas, though no text was copied. Common
    parameter names (`amount`, `from_currency`, `to_currency`; `value`, `from_unit`, `to_unit`),
    with the JSON between two tools, made the same 13 tokens. So did a description opening with
    "Look up the". The names were changed, and no pair of tools overlaps any more.
  - 518 prompts (seed 20260930): 295 fit, for 250 rows at an expected 0.85, and 223 decline,
    for 200 at 0.9. None touches the battery (`data/sft/rev2_tool_prompts_manifest.json`).
  - `reasoning_pilot.py generate` sends an item's tools and streams the reply, as the battery
    does for BFCL. The stream reassembler dropped `reasoning_content`; it keeps it now.
- **Replay, code and SWE.** The base's answers, sampled as in the pilot (temperature 0.6, top-p
  0.95, top-k 20), one per prompt. **The checks, built 2026-10-02** (`sftgen/replay_verify.py`):
  - every pool: the reply finished, with reasoning and an answer, and its answer holds no think
    tag;
  - GSM8K: the final number is the gold one (decision 5's proposal; the match is recorded either
    way, so the other choice is a filter at assembly);
  - OpenCoder: the reply's code passes every test, in the battery's sandbox. The prompts show
    their first test (above);
  - SciRIFF: where the prompt says "Only output the JSON object" (its NER tasks, 115 of 337
    prompts), the answer is that JSON alone. A reply that wraps it in a fence or a sentence would
    teach the model to break an explicit format instruction, which IFEval measures;
  - oasst1, Aya and SWE-Swiss: finishing is the check. SWE-Swiss has no tests.

**The single-turn generation, built 2026-10-02:** the SQL pool, the tool rows, code and replay,
4,326 prompts, one window or several (`patches/rev2_generate_window.sh`):
- **The bare base answers**, thinking on, through `reasoning_pilot.py generate`.
- **No reply runs past what its row can hold** (`--block 8192`). Before each request the server
  renders the prompt with its own template, tools included, and counts it. The reply may then
  take the block less the prompt, plus 16 tokens of slack, since the row is rendered again for
  training. Every pool drops rows over the block anyway; the cap stops the replies that loop. In
  the mini-battery's calibration, 7 of 120 BFCL replies looped to 12,288 tokens and took 43.5%
  of the pass's tokens.
- **Code replies may not fit.** In the same calibration, a fifth of the base's closed HumanEval+
  replies ran past 8,192 tokens (26 of 145, 31 of 147). If OpenCoder's and SWE-Swiss's replies
  run as long, the cap cuts them, and the code rows keep only the shorter ones: data that could
  teach shorter code reasoning. The window's counts will show how many, and the mini-battery's
  brevity check compares a checkpoint's reasoning length on HumanEval+ with the base's.
- **Rows are written whole, as they finish**, in one write each. A window's time limit stops a
  step mid-run, and the next window resumes it: answered items are skipped, failed ones run
  again. A checker reads one row an item (`latest_rows`), so a retried item counts once.
- **Sampling is sent explicitly.** Every Revision 2 generation so far sampled with llama-server's
  default `min_p` of 0.05, because none sent one (found 2026-10-02 on the battery server's
  `/props`): the pilots, the prefill pilot and Target C's 154 rows. Qwen recommends 0, and the
  mini-battery measures at 0. The generator now sends 0.05, so the data stays as piloted and a
  different server default can't change it (decision 8).
- **Rehearsed on the Mac** against a stand-in server: 100 items over the four pools. A pass
  stopped mid-run left 35 whole rows and resumed the other 5, with no duplicates. Every budget
  followed the rule, and the checkers read the replies, the tool rows' streamed calls and
  OpenCoder's tests in a sandbox included. Whether the real server counts the prompt as the
  rendering does, tools included, shows in the first rows of a real window: each row records
  both counts (`rendered_prompt_tokens`, `prompt_tokens`).

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
- **Tool calls keep their arguments.** Found 2026-10-01, while building the Target C pilot.
  - **The bug.** The generators record calls in the OpenAI format, with `arguments` as a JSON
    string. The template renders a call's parameters only from a mapping, so a string renders as
    `<function=run_sql>\n</function>`, a call with no arguments. llama-server parses the string
    into an object before applying the template (`func_args_not_string`), so serving shows the
    model its parameters. The HF tokenizer doesn't.
  - **What Gate 2 trained on.** Decoding its training tokens: all 6,061 real tool calls were
    empty, `add_and_execute_jupyter_code_cell` 2,561, `run_sql` 1,851, `run_python` 760,
    `final_answer` 557 and `finish` 332. The 888 calls with parameters are all the template's own
    format example, `example_function_name`.
  - **The fix.** `thinking_record` now parses the arguments as the server does, and rejects a row
    whose arguments aren't a JSON object. **Built 2026-10-01**
    (`tokenize_masked.tool_arguments_as_objects`). It covers every Revision 2 row with tool
    calls: Target C and the tool rows.
  - **Booleans.** Found in the Target C pilot. The template prints a top-level argument that isn't
    a mapping or a list with Python's `str`, so `true` trains as `True` and `null` as `None`.
    Served, the grammar holds the model to JSON, so it wrote `true`. A boolean or null now passes
    as its JSON text (`_as_written`). 30 of the tool rows' 295 fit prompts call a tool with a
    boolean parameter.
- **Rows in a block see each other, unless the model is told where they end.** Full attention,
  the GatedDeltaNet state and its 4-tap convolution all carry from one row into the next. Found
  in the code on 2026-09-29. **Built and checked 2026-10-02**
  (`reports/gate-evals/20261002-packed-rows-check.md`):
  - **The dataset** lists each block's segments, `seq_lens`: each row with its separator, then
    the padding (`pack_thinking`, `build_masked_dataset`).
  - **The collator** turns them into the model's packed-sequence arguments
    (`patches/recipe-train-packed-rows.patch`, `patches/packing/packed_rows.py`). Position ids
    restart at each row, as each conversation's do when it is served, and `cu_seq_lens` mark the
    boundaries. The batch stays one block.
  - **Traced in the box's code:**
    - with an all-ones mask, flash attention takes its variable-length path;
    - the GatedDeltaNet rule gets the boundaries as `cu_seqlens`;
    - the convolution runs transformers' torch fallback, which ignores them (the box has no
      `causal-conv1d`), so it now goes to FLA's `causal_conv1d` when they are given;
    - the loss ignores the extra arguments.
  - **A guard:** the recipe refuses `seq_lens` under any attention but flash's. Under any other,
    rows would attend to each other again, silently.
  - **Checked on the CPU, on the box** (`patches/packing/check_collator.py`): the patched
    collator over a Revision 2 dataset of the pilot's rows, 56 blocks and 190 rows. The
    segments, positions, separators and labels all reach the model as built.
  - **Checked on the GPU** (`patches/packing/check_packed_rows.py`, a window of under 9
    minutes). A block of four real rows ran with its boundaries, without them and row by row.
    - With boundaries, each row's last hidden state equals the row alone, bit for bit.
    - The block's loss equals the rows' weighted loss, to 7 digits (0.15419).
    - Its LoRA gradients match the rows' weighted gradients within 1.2-1.7%, the rounding of a
      backward pass run at another loss scale.
    - Without boundaries, the later rows move 39-71%, the loss rises to 0.221, and the gradients
      share a cosine of only 0.25-0.46 with the rows' own.
  - **The kernels on their own** (`patches/packing/check_kernels.py`):
    - attention's variable-length path computes what the plain one does, and both are causal;
    - with boundaries, the rule is bit for bit the same, and the convolution differs only by
      rounding;
    - no gradient crosses a boundary through any of the three.

    Without boundaries, a block's first row still moves 29%: the convolution's 0.3% rounding
    difference, carried through 40 routed layers.

## Assembly

**Built 2026-10-02** (`sftgen/assemble_rev2.py`). Gate 2's assembler took finished rows from fixed
files. Revision 2's rows are the base's replies that passed their pool's check, so this one:
- **Joins each kept reply to the prompt it answered**, and makes the record the template labels:
  the prompt, then one assistant turn with the reasoning (a prefill included), the answer and any
  tool calls. Target A's rows train on their training prompt. Target C's loops are records
  already.
- **Drops a row the block can't hold**, on the server's counts: the prompt, the reply, the
  newline after it and the separator. `build_masked_dataset` counts again exactly, on the box,
  and compares (`estimates` in its report). On Target C's 154 rows (2026-10-02) the server's
  count was exact for 129 and over by 1 to 28 tokens for 25, never under. On the whole mixture
  (2026-10-05) it was under for 202 rows in three scripts, by up to 1,296 tokens ("The
  generation"); the builder counts exactly, so no row went over the block. On every row, the
  last request tokenized exactly as the server counted it; the difference is all in the last
  turn, counted as it was generated, then rendered and tokenized again for training.
- **Runs `decontaminate.py`'s gate over the whole record**, the replies included. The prompts were
  gated when they were built; the replies are new text.
- **Picks rows to the table's token budgets**, with a seeded shuffle per pool. A pool short of its
  budget gives all it has, and the shortfall is reported, never padded. A budget counts whole
  rows, prompts included, since training time is paid per token; the manifest also counts the
  base's own tokens, the part that trains.
- **Takes the owner's decisions as parameters**, each defaulting to its proposal: decision 1
  (`--scale`), 2's open question (`--target-a-doubts`), 5 (`--gsm8k`), 6 (`--share`) and 7
  (`--target-c-failed-turns`). Since 2026-10-02 decision 7 is applied: a Target C turn whose call
  failed is context, not trained (decision 7 has the numbers).

The records go through `thinking_record` with the model's own template in the tests: a plain
reply, a tool call, and Target A's training prompt. On Target C's 154 rows, the assembler counts
803,029 tokens, 378,639 of them the base's own, as the volume run counted them. No row is over
the block, the gate catches none, and 3 overlap a battery item below the line. The line is
197k tokens short of its 1.0M.

**Expected to bind: the SQL line.** SynSQL's prompts alone are 2,383 tokens at the median
(chars/3.5), and the pilot's SQL replies ran about 1,400. At about 3,800 tokens a row, 1.0M holds
some 260 rows, not the table's 500, which assumed shorter prompts. Decision 6 can move tokens to
it; the generated rows will tell.

## The generation, run 2026-10-03/05

Four windows on the box (`patches/rev2_generate_window.sh`), the bare base with thinking on and 8
slots, each window resuming the one before. The next window was armed while one ran, and started
once its server stopped. About 27 hours of GPU generated 11.9M tokens, at 109-138 a second, the
slowest on SQL's long prompts. Each pool's check:

| Pool | Replies | Kept | The rest |
|---|---:|---:|---|
| `tool_fit` | 509 (214 a top-up) | 506 | 2 cut at the budget, 1 with no call |
| `tool_decline` | 223 | 209 | 13 cut (loops), 1 that called a tool |
| SQL (SynSQL) | 1,112 | 368 | 354 right only on the dataset's tiny database, 263 wrong on it, 116 cut, 11 errors |
| OpenCoder | 800 | 744 | 32 cut, 24 failing their tests in the sandbox |
| SWE-Swiss | 211 | 208 | 3 cut |
| oasst1 | 674 | 657 | 13 cut, 4 with no reasoning or no answer |
| Aya, SciRIFF | 337 each | 329, 332 | cut; every JSON-only SciRIFF prompt got the JSON alone |
| GSM8K | 337 | 283 | 35 with another number, 19 cut |
| Target A, the other 14 cells | 252 | 222 | 12 wrong, 13 cut, 5 errors |
| Target A, ClickHouse by `recall` | 522 | 162 | 490 verify, but 328 of those doubt; 28 wrong |

- **`tool_fit` came short first.** Its rows are short, about 805 tokens, so its 293 held 236k
  of its 400k. A top-up of 214 prompts (`tool_rows generate --fit 510`: the first 295 the same,
  one repeat dropped) ran in the third window, with the owner's approval (2026-10-04).
- **Replies ran shorter than this ADR assumed for replay** (2,560 tokens): oasst1 2,055, Aya
  1,738, SciRIFF 2,213 and GSM8K 2,018 on average, its kept ones shorter. OpenCoder's ran 2,926
  and SQL's 3,158. Few were cut at the budget: OpenCoder 32 of 800, where its quota allowed for a
  fifth.
- **Target A's doubts came back more often than in the pilot.** 328 of the 490 verified recall
  traces state Sunday = 1 again after the convention (67%, against the pilot's 52%), so 162 rows
  are kept where about 260 were planned. In a sample of 14, most are the old belief back ("Wait,
  is there any chance `toDayOfWeek` returns 1 for Sunday?") and a few a contrast with other
  databases, which the filter counts too; the traces reach the right SQL only after the detour.
  Wording 2 did it in 85% of its kept traces (100 of 118), the others in 55-69%.

**At 10M (decision 1's proposal) the pools held 8.89M tokens** (`assemble_rev2`, 3,825 rows).
SQL, the tool rows, OpenCoder, SWE-Swiss and SciRIFF filled their budgets. The rest fell short:

| Line | Tokens | Budget |
|---|---:|---:|
| Target A | 681,098 | 800,000 |
| Target C | 803,029 | 1,000,000 |
| oasst1 | 1,366,990 | 1,600,000 |
| Aya | 547,807 | 800,000 |
| GSM8K | 479,979 | 800,000 |

Replay, at 3.20M, would have been 36% of the mixture rather than 40%. One OpenCoder reply was
caught by the battery gate.

**Decision 1, taken 2026-10-05: 10M, with a top-up** of the short single-turn pools before
training. The other choices were to train on the 8.89M, or a budget of about 6M, which every pool
fills. Target C stays 197k short, as its volume run left it. The top-up is sized by what the
generation measured, with a tenth more:
- **Replay, 547 prompts** (`select_prompts --exclude data/sft/rev2_prompts.jsonl --rows`): the
  shortfall over the mean kept row, a tenth more, over the measured keep rate.

  | Pool | Short | A kept row | Rows wanted | Keep rate | Prompts |
  |---|---:|---:|---:|---:|---:|
  | oasst1 | 233,010 | 2,081 | 123 | 0.975 | 127 |
  | Aya | 252,193 | 1,665 | 167 | 0.976 | 172 |
  | GSM8K | 320,021 | 1,696 | 208 | 0.840 | 248 |

  None repeats a prompt the first selection took, by id or by text. On the way, `--exclude` was
  found to let a prompt through under another id where a pool repeats it. It no longer does, and
  neither selection changes.
- **Target A, 246 plain items for `recall`** (`target_a_hints volume-items --topup`): ClickHouse's
  weekday and weekend rows from new tables (no table seed meets an earlier draw's), in wordings 0,
  1 and 3 only. 79 rows cover the shortfall. Those wordings kept 37.2% and 32.7% of the items, so
  with a tenth more that is 204 weekday and 42 weekend items. The default volume still builds
  byte for byte.
- **One window of about 3.5 hours:** Target A's plain phase, splice and recall (about 1 hour),
  then the replay prompts (about 1.1M tokens).

**The top-up, run 2026-10-05**, in one window of 3 hours 19 minutes:
- **Target A:** 233 of the 246 replies verify, and 132 of those doubt (57%, against the volume's
  67%). 101 rows are kept, 149k tokens, more than the 79 the shortfall needed. Without wording 2
  the doubts fell as planned; wording 1 still doubts most (68%).
- **Replay:** 489 of the 547 kept: oasst1 124, Aya 168, GSM8K 197 (37 with another number, 14
  cut: 79% kept, against the 84% sized for).

**The mixture, assembled 2026-10-05** (`data/sft/rev2_mixture_manifest.json`): 4,326 rows and
9,821,199 tokens, 7,660,136 of them the base's own. Every line fills its budget but Target C
(803,029 of 1,000,000). Target A takes 250 of its 263 recall rows and 216 of its 222 cell rows.

**Its blocks** (`build_masked_dataset`, on the box): 1,207 blocks of 8,192 tokens, 99.8% full, at
most 18 rows a block, 78.2% of the tokens trained, and no row too long. Against the assembler's
counts, 3,663 rows are exact and 461 fewer by up to 28: the last turn, re-rendered, as on Target
C. 202 are more, by up to 1,296: Telugu, Tamil and Thai replies in Aya (155) and oasst1 (39). The
text the base generated in those scripts tokenizes into more tokens than it generated, so there
the server's count is under; no reply holds a replacement character. In all, 9,858,792 tokens
(0.4% over) and 7,735,532 trained.

## The training and its gate, run 2026-10-05/06

**The training** (`patches/rev2_train.sh`, `~/benchlab/runs/2026-10-05-qwen36-rev2-train/`) ran
the recipe as Gate 2 ran it on the 1,207 blocks: rank 4, learning rate 1e-4, one epoch. 1,207
steps took 8 hours 15 minutes, 24.3 seconds a step. A checkpoint was saved every 100 steps, 13 GB
in all. train_loss came out at 0.1548, and it was flat from the start: 0.175 over the first 100
steps, 0.14-0.16 per 100 after. The rows are mostly the base's own replies, so there is little in
them the base doesn't already predict. A flat loss is what on-policy data gives, and it says
little about what the rows teach; the targeted tests measure that.

**The gate** (the thinking-on mini-battery, "Validation" below) ran the final adapter on 2026-10-06
in one window of 5 hours 8 minutes, against the calibration's base passes. **It passes, with no
flag:**

| Benchmark | Base | Adapter | Worse / better | p | A/A |
|---|---:|---:|---|---:|---:|
| IFEval | 88.0 | 86.0 | 12 / 8 | 0.50 | 87.5 |
| BFCL irrelevance | 81.7 | 82.5 | 2 / 3 | 1 | 78.3 |
| BIRD | 70.0 | 67.3 | 7 / 3 | 0.34 | 68.7 |
| HumanEval+ | 84.0 | 82.8 | 11 / 9 | 0.82 | 86.5 |

- No reply opens a second reasoning block.
- Reasoning that never closes: BFCL 7 → 3, HumanEval+ 18 → 22, IFEval 5 → 7, BIRD 0 → 0. None of
  these changes is significant.
- HumanEval+'s reasoning is shorter: a ratio of 0.865 to the base's, sign test p = 0.006. The
  brevity flag needs a ratio under 0.8, and its pass rate is unchanged.
- Every change is within what two draws of the base differ by.

The same checks flag Gate 2's adapter on all four benchmarks (the calibration). This one keeps
what the base does, which is what the gate asks of it. It shows no gain either, and none was expected: none of the four benchmarks asks
for what the mixture targets. The earlier checkpoints stay on the box for a dose check, should a
targeted test call for one.

**The Target A test** (`sftgen/target_a_eval.py`, built 2026-10-06) asks for the conventions
directly. The base and the adapter answer the same 228 Target A prompts, and every reply is checked
on its dialect's sandboxed engine.
- **What is held out is the prompt, not only the table.** The model sees a table's schema and the
  question, never its rows. So a new table in a training domain asks a training prompt again: the
  training drew 124 of ClickHouse's 126 weekday prompts and all 18 of its weekend ones. The test's
  tables come from six domains that no training row names (`synth.held_out_domain_names`): new
  table names, timestamp columns and row nouns, with the training domains' column shapes.
  - Checked against the mixture: none of the test's prompts is a training prompt, and none of the
    4,326 rows names a held-out table or timestamp column.
  - The question wordings are the training ones. So the test measures the conventions on unseen
    tables, and the held-out probe measures the far transfer.
- **The cells:**
  - ClickHouse's weekday and weekend cells, 36 and 24 items. The base believes Sunday = 1 there:
    its plain sampling verified 7 of 96 in the prefill pilot.
  - The 14 other cells, 12 items each, 168 in all. Target A taught ISO for ClickHouse. A model
    that learned "ISO everywhere" would now fail DuckDB's, PostgreSQL's and MySQL's weekdays,
    which the base mostly gets right.
- **Paired, on one server.** The battery's server loads the adapter, and each request names its
  scale (`reasoning_pilot generate --lora-scale`): 0 for the base, 1 for the adapter. Both states
  use the same sampling and the same seed for an item, so an item's two replies differ only in the
  adapter. A parity check first confirms that scale 0 is the bare base.
  - The sampling is the generation's (min_p 0.05), so the base can be read against its own rates
    in the generation.
  - The budget is the mini-battery's 12,288 tokens.
- **Right means right on more than one table** (`target_a_eval recheck`, added 2026-10-06 during
  the run). Every item asks for a count, and on a single table a wrong count can equal the right
  one.
  - In the base's pass, two replies counted Sunday as ClickHouse's day 1, which is Monday. They
    verified anyway: those tables hold as many Mondays as Sundays (215 and 215, 210 and 210).
  - So a reply that verifies must also return the gold SQL's count on two more tables of its
    domain. That caught those two and no other of the base's 159 verified replies.
- **Read as the gate reads:** per cell and per group, how many items each state gets right, how
  many only one of them does, and an exact McNemar test on those. Also counted: the ClickHouse
  replies that state Sunday = 1 (`target_a_hints.doubts`), and reasoning length.
- **One window of about 2.5 hours:** 456 replies at about 2,400 tokens each, at the generation's
  rate (`patches/README.md`, "The Target A test").

**The same check on the training's Target A rows** (2026-10-06) finds one wrong row in the
mixture:
- every verified reply of the generation's Target A pools was rechecked on two more tables;
- the 222 cell rows all hold;
- of the 723 verified ClickHouse `recall` replies, 4 matched by chance. Three of them doubted the
  convention, so the assembler had dropped them already;
- the fourth is in the mixture, 1 of its 250 recall rows: Thursday as `toDayOfWeek(...) = 5`,
  which is Friday.

The effect on the adapter is negligible. The fix is for later generations: their Target A replies
go through the recheck before assembly.

**The Target A test, run 2026-10-06** (`reports/gate-evals/20261006-target-a-test.md`), in one
window of 2 hours 27 minutes. Parity held: scale 0 equalled the bare base on all 8 prompts, and
scale 1 changed 4 of them.
- **The adapter has learned ClickHouse's weekday numbering.** On the target cells, the base got 10
  of 60 right and the adapter 58. 49 items were right for the adapter alone and 1 for the base alone
  (exact McNemar p = 9e-14).
  - Weekdays: 9 of 36 → 34 of 36.
  - Weekends: 1 of 24 → 24 of 24.
  - Its two misses both count Thursday as 5, MySQL's number.
- **Nothing else got worse.** On the 14 other cells, 147 of 168 → 156 (17 items for the adapter
  alone, 8 for the base alone, p = 0.11).
  - ISO didn't spread to the other dialects: their weekday and weekend items went from 58 to 61 of
    72.
  - Timezones went from 42 to 48 of 48 (p = 0.03). The base's PostgreSQL misses there were the
    direction trap.
- **The old belief survives as a doubt.** 29 of the adapter's 60 ClickHouse traces still state
  Sunday = 1, typically once, against a median of 7 times in all 60 of the base's. All 29 settle
  on ISO.
- **Reasoning keeps its length:** a median of 1,489 and 1,519 tokens, against the base's 1,545 and
  1,460.
- **Left:** DuckDB's `dayofweek` (Sunday 0) gets MySQL's numbering in about a third of its items,
  in both states. Target A trained DuckDB only on the base's verified replies, which can't fix a
  mistake the base makes. The recall prefill that fixed ClickHouse would apply there.

**The held-out probe and dsbench, run 2026-10-06/07** (`reports/gate-evals/20261007-probe-dsbench.md`).
Both suites ran through the owner's pi harness, k = 5, thinking on, in one window of 3 hours 36
minutes.
- **The weekday convention carries over to prompts unlike the training rows:**
  - the probe's weekday task goes from 0 of 5 runs to 4 of 5 (Fisher p = 0.048);
  - dsbench's `da_cancel_dow` goes from 0 of 5 to 5 of 5 (p = 0.008).

  The base writes MySQL's numbers in both.
- **The weekend carries over only in part.** On `da_weekend_delay` the adapter writes
  `IN (1, 6)` three times in five: ISO's Saturday next to the old Sunday. It passes 1 of 5, the
  base none.
- **Totals:**
  - probe 25 → 28 of 30 runs, 5 → 6 of 6 problems;
  - dsbench 94 → 103 of 115 runs, 19 → 21 of 23 problems (3 gained, 1 lost; McNemar p = 0.63).
- **Nothing significantly worse.** The one problem lost, `da_redeye_count` (5 → 2 of 5, p =
  0.17), misses on reading the population, not on a convention.

The generalisation claim ADR-004 set, that the fine-tune moves both the probe and dsbench, holds
for the dialect skill.

## After the targeted tests: the recall round (proposed 2026-10-07)

The targeted tests leave two gaps in Target A's skill. Both are mistakes the base makes and the
adapter kept. The route that fixed ClickHouse's weekdays, `recall` (the convention written into
the base's own reasoning, where its plain trace first turns to the weekday function), applies to
both:
- **ClickHouse's weekend inside a larger query.**
  - On dsbench's `da_weekend_delay`, the adapter wrote `IN (1, 6)` three times in five: ISO's
    Saturday beside the old Sunday = 1.
  - Revision 2's weekend rows were one shape, a bare count, over 18 prompts, and the adapter got
    every such item right in the Target A test.
  - So the round adds two shapes (`conventions.WeekendFiltered`, `Workweek`):
    - the weekend beside a category filter;
    - its complement, Monday to Friday. That range is 1-5 in ISO and in Sunday = 0, and the old
      belief moves the whole range to 2-6.
- **DuckDB's `dayofweek`** numbers Sunday 0, and `isodow` is ISO.
  - In the Target A test, 19 of the base's 24 DuckDB weekday and weekend traces call `dayofweek`
    "1 for Sunday, 7 for Saturday", MySQL's numbering. 8 of its 9 wrong answers follow from it.
  - The adapter is no better (16 of 24), because Revision 2 trained DuckDB only on the base's own
    verified replies.
  - DuckDB gets its own three `recall` sentences (`target_a_hints.DUCKDB_RECALL_WORDINGS`). They
    name `dayofweek`, `EXTRACT(DOW ...)` and `isodow`, and every claim is checked on the engine.
  - A DuckDB trace that states Sunday = 1 now counts as a doubt, as a ClickHouse one does.

**The items** (`target_a_hints recall-round-items`, `data/sft/target_a_recall_round_manifest.json`)
are 468 plain items from the training domains, each (dialect, family) its own draw:

| Dialect | Family | Items | Prompts |
|---|---|---:|---:|
| ClickHouse | weekend-flag | 36 | 15 |
| ClickHouse | weekend-filtered | 120 | 58 |
| ClickHouse | workweek | 60 | 17 |
| DuckDB | weekday-numbering | 120 | 78 |
| DuckDB | weekend-flag | 36 | 18 |
| DuckDB | weekend-filtered | 60 | 40 |
| DuckDB | workweek | 36 | 17 |

- **What is excluded:**
  - none is a Target A test prompt (those are on held-out domains);
  - none fails the battery gate;
  - no table seed meets an earlier draw's. The round's seed sits past them all; the date itself
    was the top-up's weekend seed.
- **The sizing:** Revision 2's recall kept about 40% of its items once the doubting traces were
  dropped, so 468 items aim at about 85 ClickHouse weekend rows and 100 DuckDB rows. ClickHouse
  draws only the top-up's wordings.
- **The checks:** `target_a_hints verify`, then `target_a_eval recheck` on two more tables. The
  assembler now drops a reply the recheck didn't hold.
- **The window:** one generation window of about 2 hours, the bare base, the plan `rr_plain
  splice:rr_plain:rr_recall rr_recall` (`patches/README.md`, "The recall round").

**A false cut, found during the run (2026-10-07).**
- **What happened:** the cut's pattern took the prose "on a weekday (Monday to Friday)" for
  MySQL's `WEEKDAY(`. The workweek prompts make the base restate that phrase in its first
  sentence.
- **The effect:** 71 of the 96 workweek items (41 ClickHouse, 30 DuckDB) were cut there, before the
  trace had turned to any function. In them the convention opened the trace. That is a `start`
  prefill, the placement decision 2 turned down. Once it even followed a heading, "**Understand
  User Goal**:".
- **The fix:** `prefill._FUNCTION` now counts `WEEKDAY(` only as code. Re-cut from the same plain
  replies:
  - the 71 fall on a real function (`toDayOfWeek`, `dayOfWeek`, `dow`);
  - none of the other 397 moves, and none loses its cut.
- **The second window:** `prefill recut` splits the splice so that the 397's replies stand. The 71
  are answered again in a short second window, `rr_recall_ww`. Their first replies are not used.
- **Revision 2 is not affected:** its 768 recall cuts all fell on `dayOfWeek` or `toDayOfWeek`.

**The recall round, run 2026-10-07** (`reports/gate-evals/20261007-target-a-recall-round.md`).
- **180 rows train, from 468 items:** ClickHouse 78 (45 weekend-filtered, 14 weekend-flag, 19
  workweek) and DuckDB 102 (53 weekday-numbering, 15 weekend-filtered, 15 weekend-flag, 19
  workweek). The round was sized for about 85 and 100.
- **The checks:**
  - 429 of 468 replies verify, and the recheck fails none of them;
  - 249 of the 429 (58%) doubt, as in Revision 2's top-up (57%);
  - all 36 wrong replies are the old belief overruling the convention.
- **DuckDB's belief resists more:** it overrules the convention in 33 of 252 replies, against 3 of
  216 for ClickHouse.
- **The false cut compared the placements, on the same 71 prompts:**
  - the convention first kept 44 rows, at a median of 1,012 reasoning tokens;
  - the recall cut kept 28, at 1,470.

  That is the prefill pilot's result again: the short rows stay out, as decision 2 has it.

**How the rows train (proposed 2026-10-07; the owner's decision).**
1. **Add them to Revision 2's mixture and open Target A's budget.**
   - Target A takes all 665 eligible rows (1.16M tokens). Those are Revision 2's 466, the 19 its
     budget left out, and the round's 180.
   - Every other pool keeps exactly Revision 2's rows: the preview reproduced Revision 2 bit for
     bit, then added the round. The mixture is 4,525 rows and 10.17M tokens (+3.6%).
   - Kept at 800,000 tokens, the round would push out, at random, about a third of the Target A
     rows that gave Revision 2's gains.
   - It needs one assembler option (`--bucket-tokens target_a=N`), so that the defaults still
     rebuild Revision 2.
   - The alternative is the round as a pool of its own: exactly Revision 2's 466 plus the 180, at
     the cost of more code. The 19 rows it would leave out are of kinds the adapter already gets
     right.
2. **Retrain from the base** with Revision 2's recipe, not continuing the adapter. That keeps the
   data on-policy, the base's own replies, and it is about 8.5 hours.
3. **Measure it as Revision 2 was measured**, about 20 hours of GPU in all:
   - the mini-battery gate (about 5 hours);
   - the Target A test, extended with held-out items of the two new shapes. The test asks
     weekend-flag and weekday-numbering; weekend-filtered and workweek need new items on the
     held-out domains;
   - the probe and dsbench (3.6 hours): `da_weekend_delay` is the problem the round is for.

**Revision 2.1, steps 1 and 3 prepared 2026-10-07.** The owner took the proposal: open the line,
and build the test of the new shapes. Runbooks are in `patches/README.md`, "Revision 2.1" and "The
recall round's test".
- **The mixture** (`data/sft/rev2_1_mixture_manifest.json`) uses `assemble_rev2 --bucket-tokens
  target_a=1200000`. The defaults still rebuild Revision 2 bit for bit.
  - It holds 4,525 rows and 10,174,379 tokens. Target A has 665 rows; every other pool is
    Revision 2's, row for row by id.
  - Its blocks are on the box (`~/benchlab/runs/2026-10-07-qwen36-rev2-1-train/`): 1,251 of 8,192
    tokens (Revision 2: 1,207), 99.7% full, 8.07M tokens trained. At Revision 2's 24.3 seconds a
    step, that is about 8 hours 27 minutes.
- **The round's test** (`target_a_eval round-items`, `data/sft/target_a_rr_test_manifest.json`)
  has 204 items on the held-out domains.
  - **Target cells (156 items):** ClickHouse weekend-filtered and workweek, 24 each; DuckDB
    weekday-numbering 36, and its three weekend families, 24 each.
  - **Guard cells (48 items):** PostgreSQL's and MySQL's weekend-filtered and workweek, 12 each.
    DuckDB's rows teach `dayofweek` = Sunday 0, and MySQL's `DAYOFWEEK`, spelled the same,
    counts Sunday 1. A model that carried DuckDB's numbering over would fail MySQL here.
  - **The checks:** no test prompt is a prompt of the mixture, and no mixture row names a held-out
    table or column. The test's tables, recheck tables included, meet none of the Target A
    test's.
- **The test runs once per adapter, against the base.**
  - **Revision 2's adapter can run it now:** how far does Revision 2 already get on the new shapes?
    dsbench's `da_weekend_delay` says not far, but that was one problem, k = 5.
  - **Revision 2.1's adapter runs it after its gate.**
  - Revision 2 against Revision 2.1 then reads the round's effect, and the two base runs check each
    other across windows.

**Revision 2.1's training and gate, run 2026-10-07/08** (`reports/gate-evals/20261008-rev2-1-gate.md`).
- **The training:** 1,251 steps in 8 hours 37 minutes; train_loss 0.1553 against Revision 2's
  0.1548. The mean loss per 100 steps stayed between 0.148 and 0.172.
- **The gate passes, with no flag and no A/A flag:** IFEval 88.0 → 87.0, BFCL irrelevance 81.7 →
  79.2, BIRD 70.0 → 68.7, HumanEval+ 84.0 → 87.7. Every change is within the base's own noise.
- **Revision 2's shorter HumanEval+ reasoning is gone.** Revision 2's ratio was 0.865 (sign test
  p = 0.006, under the 0.8 flag); Revision 2.1's is 0.945 (p = 0.74).
- **The round's 199 Target A rows moved nothing the gate measures,** and the gate measures none of
  what they teach. The round's test, the Target A test, the probe and dsbench follow, queued behind
  the gate.

## Budget and time

Estimated from the pilots' throughput, and measured where it says so (updated 2026-10-05).
- **Generation**, with production stopped: **run 2026-10-03/05**, about 27 hours of box time in
  four windows ("The generation" above), at 109-138 tokens a second on 8 slots:
  - SQL 9.0 hours, code 6.3, replay 7.2, Target A 3.4, the tool rows 1.1;
  - Target C: its 2026-10-01/02 window (154 rows, 0.80M tokens);
  - the top-up (decision 1): about 3.5 hours more.
- **Training:** **run 2026-10-05/06** on the mixture's 1,207 blocks, 8 hours 15 minutes: 24.3
  seconds a step, about the packed rows' warm step (23.6 s, 2026-10-02). It was planned at 9-15
  hours, the upper end from the 182 tokens a second the seq-4096 report measured for 8,192-token
  rows.
- **Checks:** a checkpoint's mini-battery takes one window of about 5 hours (the final adapter's,
  5 hours 8 minutes). The Target A test takes one of about 2.5. The full battery with thinking on
  takes about 18 hours a state for four of its eight benchmarks, and more for the other four
  (decision 9).

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

  **Built 2026-10-02** (`battery/mini.py`), and calibrated the same day
  (`reports/gate-evals/20261002-mini-battery-calibration.md`).
  - **The subsets** come from the battery's pinned items, drawn with a fixed seed:
    - IFEval: 200 of 541;
    - BFCL: 120 of the 240 irrelevance items;
    - BIRD: 150 of 1,534, in proportion to its difficulty levels;
    - HumanEval+: all 164.
  - **Thinking is on,** with Qwen's sampling for thinking mode, in a budget of 12,288 tokens (pi's
    reply cap). Greedy decoding with thinking on loops, so the passes sample.
  - **Seeds.** Every state draws the same seed for an item, so base and checkpoint start from the
    same random numbers. The A/A pass draws other seeds, so its flips are the full sampling noise.
  - **Compared per item, against the base:**
    - the passes;
    - BIRD's SQL that SQLite can't run;
    - reasoning that never closed, and replies cut at the budget;
    - a second reasoning block inside the reply;
    - reasoning length.
  - **A flag** is a significant change for the worse (exact McNemar at 5%). For length, it takes a
    sign test and a ratio under 0.8 of the base's.
  - **The sizes are set for Gate 2's habits, not for small pass-rate changes.**
    - BFCL irrelevance fell 22 points.
    - BIRD's SQLite errors went from 47 to 176.
    - IFEval's 4.6 points needed all 541 items. The brevity check catches the same habit sooner.
    - The full battery stays the final check.
  - **The calibration** runs the base twice on these items. It measures:
    - the cost of an item with reasoning;
    - how often the base's reasoning closes within the budget;
    - the noise.

    The sizes and the budget are confirmed after it.

    **Run 2026-10-02: the sizes and the budget stay.**
    - A state's four passes take 4.5-4.7 hours, so a checkpoint's gate is one window of about 5.
    - Thinking moves the base both ways, against Gate 2's thinking-off base on the same items:
      IFEval +9.0 points, BIRD +9.3, BFCL irrelevance −10.8, HumanEval+ −6.8. The gate compares
      a checkpoint with this thinking-on base.
    - The base never closes its reasoning on 10-11% of HumanEval+'s items, 6-8% of BFCL's, 2.5%
      of IFEval's and 0-1% of BIRD's. BFCL's scorer passes such a reply, which answers nothing,
      so a pass now needs closed reasoning.
    - Two draws of the base disagree on 8.5-16% of the items. A checkpoint's passes must move
      5.8-8.8 points to flag at 80% power. The A/A pass flags nothing.
    - Run on Gate 2's own passes, the checks flag its adapter on all four benchmarks, and only
      BFCL at half scale, as Gate 2's dose check found.
    - At 8,192 tokens, a fifth of HumanEval+'s closed replies would no longer close; 12,288
      stays.
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

**Taken.** On 2026-10-03 the owner took the proposals of decisions 2-8, and on 2026-10-05
decision 1: 10M, with a top-up ("The generation" above). Decision 9 is open.

1. **Budget:** 10M tokens (9-15 hours of training, and 25-35 more of generation: "Budget and
   time"), or more. **Taken 2026-10-05:** 10M. The pools held 8.89M of it, so the short ones are
   topped up before training.
2. **Target A's reasoning.** The hint pilot ruled out the hint-conditioned base as designed: only
   4-11% of its traces didn't cite the hint. The prefill pilot (2026-10-01) found the route:
   - the base's own plain trace, cut where it first turns to the weekday function, the
     convention written there, and the base continuing (`recall`);
   - 95 of 96 replies kept, at the base's length, against 7 of 96 for plain sampling.

   Proposed: `recall` for ClickHouse's weekday and weekend rows, and no teacher (its volume is
   built, "Where the reasoning comes from"). Also to decide:
   whether to drop the traces that raise Sunday = 1 again before settling on ISO, about half.
   Proposed: drop them; it costs an hour of box time (the assembler's `--target-a-doubts`).
3. **SQL:** SynSQL-2.5M alone (proposed), or also BIRD and Spider train under CC BY-SA 4.0.
4. **jupyter-agent (559 rows in Gate 2) and DataMind (640):** their answers are other pipelines'
   text, which principle 2 excludes. Drop them (proposed: Target C, the SQL pool and the tool rows
   carry the DS and tool behaviour), or regenerate them with the base as the agent, which needs
   their data in the sandbox.
5. **Replay prompts:**
   - WildChat: clean-origin sets only (proposed), or also WildChat's first user turns, which are
     human-written; their GPT-4 answers would be replaced.
   - Unlisted licences. The Tulu card lists FLAN v2's licence as unspecified, and its tasks come
     from many source datasets, each under its own licence. SciRIFF's card lists none for 19 of
     the sample's tasks (4,824 rows). Keep them, or leave both out of anything redistributable
     (proposed: the other replay sets have prompts to spare). FLAN v2 stays on the allowlist, as
     Gate 2 had it, and SciRIFF's unlisted tasks stay out, until this is decided.
   - GSM8K's gold answers: keep only the base's correct replies, as a verified pool does
     (proposed: the check is free, and a wrong solution would train at full weight), or every
     reply that finishes, as the rest of replay does.
6. **The proportions above**, and within the buckets (the selector's defaults, proposed):
   replay 0.4 oasst1, 0.2 each Aya, SciRIFF and GSM8K, FLAN v2 0 until decision 5; code 0.75
   opencoder, 0.25 SWE-Swiss.
7. **Target C's failed turns.** In the pilot, about 28% of a passing loop's labelled text sits in
   turns whose tool call failed. The commonest failure is a wrong guess at the ClickHouse client's
   API.
   - Proposed: no loss on those turns; they stay in the row as context, so the fix that follows
     trains with its cause in view. The oracle checks the delivered table, not each step.
   - Or: train every turn, as the loop happened.

   Both are built (2026-10-02): the assembler marks a failed turn `"loss": False` by default, and
   `--target-c-failed-turns train` trains it. On the 154 rows, 141 turns in 88 rows are marked,
   and the base's own tokens fall from 378,639 to 305,754; the rows' tokens don't change.

   Also: Target C at 8,192 tokens with a quota per family (proposed), or a 16,384-token step for
   its rows, which would keep the loops that reason through credit_leak's leak. The volume run
   measured both: 154 rows and 0.80M tokens at 8,192; 247 rows and 1.87M tokens at 16,384, with
   58 of credit_leak's 71 loops instead of 22. Its rows are already generated. In the kept rows,
   20.5% of the assistant text sits in turns whose call failed.
8. **The data's sampling.** Every Revision 2 row so far was sampled with `min_p` 0.05,
   llama-server's default, while Qwen recommends 0 and the mini-battery measures at 0.
   - Proposed: keep 0.05 for the rest of the generation. Target C's 154 rows and the pilots'
     yields and lengths were measured with it, and the base's distribution with its least likely
     tokens trimmed is still its own.
   - Or: 0, as Qwen recommends, for the rows not yet generated. Target C's rows stay as they are.
9. **The full battery with thinking on.** At the mini-battery calibration's cost per item,
   IFEval, BFCL, BIRD and HumanEval+ at full size take about 18 hours a state. MMLU-Pro, GPQA,
   LiveCodeBench and DS-1000 weren't measured with thinking on, and add to that; with thinking off
   and 4,096 tokens, Gate 2's base already hit the limit on half of LiveCodeBench's items.
   - Proposed: the base once, since every candidate compares with it, then only the checkpoint
     the mini-battery passes. Each in resumable windows (`MAX_HOURS`), the base's first window
     measuring the four unmeasured benchmarks' cost before the rest is planned.
   - Or: thinking-on subsets of those four, sized like the mini-battery's, and the full battery
     for the final candidate only.

   **Built 2026-10-02 for either** (`battery/full.py`; the base's run directory is prepared on
   the box):
   - every pinned item asks for thinking within 12,288 tokens, each benchmark in a seeded order.
     A window step `bench:state:N` answers the first N, a random sample, and a later step
     resumes the pass;
   - DS-1000's stop strings are dropped: llama-server matches them against the reasoning too,
     and DS-1000's extraction cuts at the same markers itself;
   - `report.py` counts a thinking-on pass as the mini-battery does: a reply that never closed
     fails, and a pass counts only the items it answered. On Gate 2's run it gives the same
     report, byte for byte.

   The base's run needs no retrain, so it can use the box while the other decisions wait.

## Action items (Revision 2)

1. [x] Rendering and packing. **Done 2026-09-29; the last point 2026-10-02**
   (`reports/gate-evals/20260929-thinking-rendering-packing.md`,
   `reports/gate-evals/20261002-packed-rows-check.md`):
   - [x] the `<think>\n` opener stays in the prompt;
   - [x] no labels before the last user message;
   - [x] bin-packing into 8,192-token blocks without splitting rows (`build_masked_dataset.py` now
     defaults to 8192);
   - [x] tests on rendered examples, with the model's own chat template;
   - [x] found on the way: the tokenizer now splits text as llama.cpp does;
   - [x] found 2026-10-01: tool-call arguments render as the server renders them. Gate 2 trained
     every one of its 6,061 tool calls with no arguments;
   - [x] found 2026-10-01 in the Target C pilot: a boolean or null argument renders as the JSON
     the model wrote, not Python's `True` or `None`;
   - [x] rows in a block see each other: a GPU check and a collator patch that passes row lengths.
     **Done 2026-10-02:**
     - the dataset's `seq_lens`, the collator patch, and boundaries for the convolution;
     - checks of the collator (CPU), the model (GPU) and the kernels (GPU);
     - packed, each row trains exactly as if it were alone.
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
   - [x] hint-conditioned generation with its hint-citation filter, piloted on 48 rows per
     dialect. **Done 2026-09-30** (`reports/gate-evals/20260930-target-a-hints-pilot.md`):
     - the hint fixes the answers, but only 4-11% of hinted traces don't cite it;
     - the gap is ClickHouse's weekday numbering alone;
     - the timezone family, measured again, verifies 44 of 48.
   - [x] a source of ClickHouse weekday reasoning: a reasoning prefill or plain sampling at
     scale; a teacher if neither yields. **Done 2026-10-01**
     (`reports/gate-evals/20261001-target-a-prefill-pilot.md`):
     - the `recall` prefill keeps 95 of 96 at the base's length;
     - `start` keeps 92 of 96, at two-thirds of the base's length;
     - plain sampling keeps 7 of 96.
   - [x] the volume's items and tooling. **Built 2026-10-02:** 4 checked wordings of the
     sentence; the plain phase stopped at 512 tokens; the doubt filter, recorded by `verify` and
     applied by the assembler; rows spread over the prompts; the window's `splice` step.
   - [x] generate Target A's rows in a window (`ta_cells ta_plain splice:ta_plain:ta_recall
     ta_recall`). **Run 2026-10-04/05:** 222 of the 252 cell rows and 162 of the 522 recall rows
     kept; 328 recall traces doubt ("The generation" above).
   - [x] its top-up: 246 recall items without wording 2. **Run 2026-10-05:** 101 rows kept
     ("The generation" above).
3. [x] SQL: acquire SynSQL-2.5M's databases, build the prompts from them, write the
   execution-match verifier. **Done 2026-09-30:** 1,112 prompts, 0 battery overlaps. SynSQL's
   databases hold about two rows a table, so the check also runs every query on three bigger
   variants of each. Generation waits for the GPU window with the rest; its window is built
   (item 6).
4. [x] Tool rows: generators for tool lists, gold calls and requests (fitting and not), and the
   gold-call checker. **Done 2026-09-30:** 518 prompts, 0 battery overlaps; generation waits for
   the GPU window with the rest; its window is built (item 6).
5. [ ] Target C: the agentic pilot (about 20 tasks, thinking on) for yield and length, then volume.
   - [x] the pilot. **Run 2026-10-01** (`reports/gate-evals/20261001-target-c-agentic-pilot.md`):
     20 of 21 pass, 14 fit 8,192 tokens, every turn has reasoning, no teacher needed.
   - [x] a quota mode in the generator. **Built 2026-10-01**:
     - each family runs until its rows fill, with the length taken from the server's counts;
     - each dataset is gated by its own oracle;
     - a run can be resumed;
     - a run that names an answer key is never kept.
   - [x] the agent's own ClickHouse login (ADR-003 §7). **Done 2026-10-01**: the key is out of
     its reach, checked live and by `ml_tasks`' oracle gate.
   - [x] a per-turn mark for decision 7. **Built 2026-10-02:**
     - `thinking_record` leaves a turn marked `"loss": False` unlabelled, and still checks it;
     - `agentic.tools.is_error` reads a failed call as the executors report it. On the 154 kept
       rows it finds the volume report's 141 of 1,158 turns, 20.5% of the assistant text;
     - the assembler applies it (`--target-c-failed-turns`).
   - [x] the window scripts. **Built 2026-10-01** (`patches/target_c_volume_*.sh`): a 5-hour hold
     whose deadline the Mac reads, early ends, a second window that resumes the first; rehearsed
     against a stand-in.
   - [x] the volume run. **Run 2026-10-01/02**
     (`reports/gate-evals/20261001-target-c-volume-run.md`): 154 rows and 0.80M tokens at 8,192.
     275 of 282 runs passed; 121 passing loops ran over the block.
6. [ ] Replay, code and SWE: drop No Robots from the Tulu allowlist (**done 2026-09-29**),
   register GSM8K (**done 2026-09-30**), acquire Aya and SciRIFF (**done 2026-09-30**),
   select the prompts (**done 2026-09-30**; 2,696 prompts since OpenCoder's quota rose on
   2026-10-03; rerun after decisions 5 and 6),
   generate.
   - [x] the checks and the generation window for the single-turn pools (SQL, tools, code,
     replay). **Built 2026-10-02:** `replay_verify.py`; `reasoning_pilot.py generate --block`;
     `patches/rev2_generate_window.sh`; rehearsed against a stand-in. OpenCoder's prompts now show
     their first test, and the dataset's own answers pass 748 of 750 items.
   - [x] the generation windows: 4,326 prompts. **Run 2026-10-03/05:** about 27 hours over four
     windows, and 214 `tool_fit` prompts topped up in the third ("The generation" above).
   - [x] the top-up window (decision 1): 547 replay prompts and Target A's 246 items. **Run
     2026-10-05**, 3 hours 19 minutes: 489 and 101 rows kept.
7. [x] Decontamination: extend `decontaminate.py` with the battery's 13-gram index. **Done
   2026-09-29**, with short items matched whole and BFCL's schemas indexed
   (`reports/gate-evals/20260929-battery-decontamination.md`).
8. [ ] The thinking-on mini-battery, and the full battery re-baselined for the base.
   - [x] the mini-battery. **Built 2026-10-02** (`battery/mini.py`, `battery/generate.py`'s
     thinking items, `patches/battery_window.sh`'s production guard and time limit). The
     subsets are prepared on the box and match the local ones byte for byte.
   - [x] its calibration. **Done 2026-10-02**
     (`reports/gate-evals/20261002-mini-battery-calibration.md`): two windows, 9 hours and 11
     minutes in all; the sizes and the 12,288-token budget stay ("Validation" above).
   - [ ] the full battery, re-baselined with thinking on: about 18 hours a state for the four
     benchmarks the calibration measured, before the other four (decision 9). Its items, the
     report's rules and the window's sample step are built (2026-10-02); the base's run
     directory is prepared on the box.
9. [x] Assemble, train (rank 4, 8,192 tokens), and gate checkpoints on the mini-battery.
   - [x] the assembler. **Built 2026-10-02** (`sftgen/assemble_rev2.py`, "Assembly" above).
   - [x] assemble, once the pools are generated and checked; then `build_masked_dataset` on the
     box, whose exact counts check the assembler's. **Done 2026-10-05:** 4,326 rows, 9.82M
     tokens, 1,207 blocks ("The generation" above). Previewed before the top-up at 8.89M. **Checked on Target C's 154 rows
     2026-10-02** (`~/benchlab/runs/2026-10-02-rev2-assembly-check/`), the only pool generated so
     far:
     - no row too long. The server's counts are exact or over ("Assembly" above): 802,944 tokens
       against 803,029, and 305,669 trained against 305,754;
     - 113 blocks, at most 2 rows each, 86.8% full: Target C's rows average about 5,200 tokens,
       so they pack poorly;
     - the builder now makes this check on every mixture: each record carries the counts the
       assembler selected it by (`meta.mix_tokens`, `meta.mix_reply`).
   - [x] the training window and the checkpoint gate's scripts. **Built 2026-10-02**
     (`patches/rev2_train.sh`, `patches/mini-battery/checkpoint.sh`):
     - the recipe as Gate 2 ran it (rank 4, learning rate 1e-4, one epoch, a checkpoint every
       100 steps), on the packed blocks;
     - the packed-rows patch applies cleanly to the box's recipe (checked, not applied: the
       recipe is shared);
     - found: the recipe keeps only its last 5 checkpoints, so the early ones would be deleted
       before the gate could read them. `recipe-train-save-limit.patch` keeps them all, about
       0.9 GB each;
     - a checkpoint's gate: its adapter exported to a GGUF, its own run directory linking the
       calibration's base passes, and one battery window for its `adapter` passes.
   - [x] train, and gate the checkpoints. **Trained 2026-10-05/06**, 1,207 steps in 8 hours 15
     minutes; **the final adapter passed its gate 2026-10-06**, with no flag ("The training and
     its gate" above). The checkpoints every 100 steps stay for a dose check, should a targeted
     test call for one.
10. [ ] The targeted tests, with the adapter against the base.
    - [x] the Target A test. **Built 2026-10-06** (`sftgen/target_a_eval.py`, held-out domains
      in `sftgen/synth.py`, `--lora-scale` in `reasoning_pilot generate`, steps `NAME:STATE` in
      `patches/rev2_generate_window.sh`): 228 items
      (`data/sft/rev2_target_a_test_manifest.json`).
    - [x] its window, the check on the Mac and the comparison. **Run 2026-10-06**, 2 hours 27
      minutes: ClickHouse's weekdays and weekends 10 → 58 of 60, the other cells 147 → 156 of
      168 ("The training and its gate" above). The check now reruns every verified reply on two
      more tables (`target_a_eval recheck`).
    - [x] the probe's and dsbench's run, base against the adapter. **Built 2026-10-06**
      (`patches/rev2_probe_mac.sh` with a battery window's hold, `agentic/compare_runs.py`):
      - pi's own settings for the model, checked against a stand-in server: thinking on
        (`enable_thinking`), temperature 0, 12,288 tokens a reply;
      - the state set on the server before each suite;
      - the probe's oracle gate passes, 6 of 6, and dsbench's, 23 of 23.
    - [x] the held-out probe and dsbench (k = 5), through pi with the 12,288-token reply cap.
      **Run 2026-10-06/07**, 3 hours 36 minutes: the probe 25 → 28 of 30 runs, its weekday task
      0 → 4 of 5; dsbench 94 → 103 of 115, `da_cancel_dow` 0 → 5 of 5 ("The training and its
      gate" above).
11. [ ] The recall round: ClickHouse's weekend shapes and DuckDB ("After the targeted tests").
    - [x] the items and tooling. **Built 2026-10-07**: two weekend families, DuckDB's `recall`
      sentences, `recall-round-items` (468 items), DuckDB's doubts counted, and the assembler
      dropping a reply the recheck didn't hold.
    - [x] the generation window (about 2 hours), then the checks on the Mac, with the recheck.
      **Run 2026-10-07**, two windows (2 hours, then 14 minutes for the 71 workweek items the
      false cut had cut at their first sentence; PR #49): 180 rows train, ClickHouse 78 and DuckDB
      102 (`reports/gate-evals/20261007-target-a-recall-round.md`).
    - [x] how the rows train: the owner took the proposal 2026-10-07 (Revision 2.1: Revision
      2's mixture with Target A's line opened, retrained from the base).
12. [ ] Revision 2.1 ("The recall round", "Revision 2.1, steps 1 and 3 prepared").
    - [x] the mixture and its blocks. **Built 2026-10-07**: `assemble_rev2 --bucket-tokens`,
      4,525 rows, 10.17M tokens, 1,251 blocks on the box.
    - [x] the round's test. **Built 2026-10-07**: `target_a_eval round-items`, 204 items (156
      target, 48 guard), staged for Revision 2's adapter.
    - [x] the round's test on Revision 2's adapter. **Run 2026-10-07**, 1 h 54 min
      (`reports/gate-evals/20261007-recall-round-test-rev2.md`).
      - Revision 2 already writes ClickHouse's two new shapes: 2 → 47 of 48. So dsbench's
        `da_weekend_delay` failure belongs to the agent context, not to a missing shape.
      - DuckDB stays the gap: 73 → 85 of 108, no cell significant.
      - The guards: 47 → 46 of 48 (PostgreSQL lost 2 items to MySQL's weekend numbers).
    - [x] the training. **Run 2026-10-07/08**, armed behind the test: 1,251 steps in 8 h 37 min,
      train_loss 0.1553 (Revision 2: 0.1548).
    - [x] the gate (the mini-battery). **Passed 2026-10-08**, no flags (5 h 24 min; launched by a
      box chain once the trainer exited rc = 0; `reports/gate-evals/20261008-rev2-1-gate.md`).
    - [ ] the round's test, the Target A test, the probe and dsbench on Revision 2.1's adapter
      (queued 2026-10-08 behind the gate).
