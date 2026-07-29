# avtext — Aviation-Text LLM Lab

## Research question

> Does LoRA/QLoRA fine-tuning a small open model (**Gemma 4 E4B**, ~4.5B
> effective params, Apache-2.0) measurably improve performance on
> aviation-specific text tasks — decoding METAR/TAF into structured data —
> versus the base model, and how does that compare to a frontier API baseline?

And, as a deployment-reality kicker: how much of any gain survives **4-bit
quantization** when the model runs locally via `llama.cpp`?

## Why this is really three learning projects

| Skill | Phase | Notes |
|---|---|---|
| A. Automated ML dataset acquisition | 1 | Immutable raw zone, checksummed manifests, dedupe keys, license bookkeeping. |
| **B. A benchmark/eval harness you can trust** | 2–3 | The heart of the project: the oracle problem, leakage, metric decomposition, statistical honesty. |
| C. Fine-tuning a small LLM properly | 4–5 | LoRA/QLoRA hyperparameters, data→behavior causality, quantization trade-offs. |

The master plan lives in [`docs/research-repo-plan.md`](docs/research-repo-plan.md).

## Method (in one breath)

Thin vertical slice first: **METAR/TAF only, ~50 stations, one task family**,
taken all the way from download → harness → baseline → fine-tune → before/after
report. Correctness comes from a **trust ladder** (round-trip + invariants →
cross-parser consensus → official gold seed → one expert pass), never from our
own aviation judgment. Test set is drawn **strictly after the model's Jan-2025
data cutoff**, with **station + time** splits enforced in code.

## Build order (each stage testable before the next)

`schema → oracles → tier-0 checks → tier-1 consensus → gold seed → eval sets → runner → scorers → stats → report` — then, and only then, fine-tune.

## Current results

_Nothing measured yet — the harness must be frozen before any training._

| Model | Task slice | Whole-record EM | Micro-F1 | Hallucination rate | Notes |
|---|---|---|---|---|---|
| Gemma 4 E4B (base, fp16) | decode→JSON (clean) | — | — | — | pending |
| Gemma 4 E4B (base, Q4)   | decode→JSON (clean) | — | — | — | pending |
| Gemma 4 E4B (fine-tuned) | decode→JSON (clean) | — | — | — | pending |
| Frontier API (reference) | decode→JSON (clean) | — | — | — | pending |

## Layout

```
src/avtext/
  ingest/     # A: fetch raw METAR/TAF/etc. from IEM & AWC
  schema/     # B: pydantic canonical decoded records
  oracles/    # B: adapters over 3 independent parsers -> canonical schema
  quality/    # B: tier-0 round-trip + invariant checks (Hypothesis)
  consensus/  # B: tier-1 field-level majority vote across oracles
  tasks/      # B: builders that freeze eval/v1/*.jsonl
  harness/    # B: runner, scorers, stats, report generator
  finetune/   # C: SFT data prep, train configs, GGUF export
configs/      # stations.yaml, eval_v1.yaml, train_run_*.yaml
data/         # gitignored bulk; only manifest.json + README committed
eval/v1/      # frozen eval sets (JSONL + content hash) — committed
reports/runs/ # per-run results JSON + markdown report — committed
docs/adr/     # architecture decision records
docs/LEARNING_LOG.md  # the running "what I learned" log
```

See [`DATA_LICENSES.md`](DATA_LICENSES.md) for per-source terms — it is part of
code review, not an afterthought.
