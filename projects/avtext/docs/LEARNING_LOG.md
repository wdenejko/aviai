# Learning log

The real output of a learning project. One entry per working session: what was
done, what was *learned* (concepts, not just tasks), and open questions.

---

## Session 1 — 2026-07-29 · Scaffold

**Done**

- Verified the target model against primary sources: `google/gemma-4-E4B-it`,
  4.5B effective params, **Jan-2025 data cutoff**, Apache-2.0. The plan's
  post-cutoff test split is correct.
- Chose the repo shape: `aviai` = uv-workspace sandbox, `projects/avtext/` =
  first subproject (ADR-002).
- Scaffolded the full skeleton: package layers with responsibility docstrings,
  configs, data-zone rules, `DATA_LICENSES.md`, Makefile, CI, ADR-001/002/003.
- Synced a lean local env (no torch/transformers — fine-tuning is cloud-side).

**Learned**

- *Verify the linchpin fact before building on it.* Gemma 4 released after this
  assistant's knowledge cutoff, so "does this model even exist / what's its
  cutoff" was a real question, not a formality — and the answer defines the
  entire eval time-split. Cheap check, load-bearing result.
- *A time split is necessary but not sufficient.* METAR/TAF are so templated that
  a base model can decode post-cutoff strings from pre-cutoff format knowledge.
  So "did fine-tuning help" is best measured on **hard cases / rare tokens /
  abstention**, not clean decodes — those risk a ceiling effect that hides the
  signal. This should shape station selection *and* eval weighting. (→ ADR-003)
- *uv workspaces* are the idiomatic answer to "umbrella repo of self-contained
  subprojects sharing one env" — one lock, one `.venv`, per-project pyprojects;
  the cost is a single shared dependency resolution.

**Addendum — companion research docs integrated** (now in `docs/research/`)

Reconciling both briefings against the plan + scaffold:

- **Reframe (→ ADR-004).** Clean decode is a solved parser problem (~99–100%);
  the measurable prize is the **messy tail + faithful briefing + abstention**.
  Headline metrics + README results table updated to lead with these; clean
  decode demoted to a reference ceiling.
- **The oracle worry is answerable** — it's the *test-oracle problem*.
  Correctness becomes a property of authoritative artifacts + machine checks,
  never my judgment: AWC/IEM ship the **decoded JSON alongside the raw** (ground
  truth largely bundled); where it isn't, it's a table lookup (Q-codes,
  contractions); for prose, check **faithfulness, not taste**. One paid expert
  pass + the **rule of three** (0 errors in 300 ⇒ <1% error rate) bounds the rest.
- **E4B is dense, not MoE** → stable fine-tuning, no router to destabilize.
- **Stack divergence is intentional:** the datasources doc sketches the day-job
  Airflow/ClickHouse/GE/dbt stack; the plan deliberately uses DuckDB + Hypothesis
  + plain scripts to keep the learning repo self-contained. We follow the plan.

**Open questions**

- Confirm exact PyPI names for the oracle parsers; weigh `metaf` (C++/MIT) as a
  cross-language 4th lineage vs its integration cost (Phase 2).

**Next session — Phase 1 (data)**

0. **Scope probe** — run 2–3 parsers over a real sample, measure the
   parser-failure rate. Validates the premise and sizes the messy-tail prize.
1. Finalize ~50 stations in `stations.yaml` (incl. AUTO-heavy).
2. `ingest/iem.py` (checksummed backfill) + `ingest/awc.py` (collector — capture
   the bundled decoded JSON too: it's a free 4th oracle voice *and* the
   post-cutoff eval pool). First raw METAR/TAF in `data/raw/` + populated manifest.
