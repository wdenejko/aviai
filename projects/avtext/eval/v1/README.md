# eval/v1 — frozen evaluation set

**Frozen 2026-07-29 · 620 records.** The `sha256` in `manifest.json` is the immutability
anchor: a run report cites it to prove exactly which bytes it scored against. Immutable by
contract — a change means a new `v2/`, never an edit in place.

Regenerate (must reproduce the same sha256, byte-for-byte):

```bash
uv run python -m avtext.harness.freeze
```

## What's in it

`eval.jsonl`, one JSON record per line:

| field | meaning |
|-------|---------|
| `id` | `sha1(station\|valid_utc\|raw)[:12]` — stable record key |
| `split` | `unseen_station` or `unseen_time` — the generalization axis |
| `station`, `valid_utc`, `raw` | the observation under test |
| `label` | `clean` / `parse_fail` / `dissent` — hard-case bucket from the trust ladder |
| `reference` | consensus field values = the scoring target (gold-validated for clean/dissent) |
| `failed`, `dissent` | which oracles failed / which fields the panel split on |

## Splits (ADR-006) — held out from future training along two axes

- **`unseen_station`** (260) — stations the model will NEVER train on:
  `KEKM EGCC KSEA EPKK`. Tests transfer to an unseen station of a *known* format family
  (each held-out station keeps same-family siblings in training, so the format is
  learnable). Carries the **decode** test: 200 clean + 60 dissent.
- **`unseen_time`** (360) — training stations, but only reports on/after `2026-05-01`.
  Tests robustness to new dates. Carries the **abstention** test: 200 clean +
  100 parse_fail + 60 dissent.

Every item post-dates `2025-02-01` — after Gemma-4-E4B's Jan-2025 cutoff, so no item the
base model could have memorized (eval/README rule 1).

## Why the two strata aren't symmetric

This is a property of the corpus, not a choice. Post-cutoff, parse failures come almost
entirely from ONE station — UUWW, at ~38% and rising — and the phenomenon is non-stationary
(YSSY, the only other candidate, had its format fixed at 2025-02 and now parses clean).
Holding UUWW out would strand the sole abstention source with nothing left to train on, so
it **stays in training** and is tested by time-holdout instead (its post-2026-05 reports,
failing at **74%**). Abstention is therefore an `unseen_time` test; `unseen_station` tests
decode + disagreement. Full reasoning in ADR-006.

## Over-weighting the tail

The messy tail is ~2% of real data; a blind sample would be 98% clean — spending the eval
budget re-checking a solved problem (ADR-004). Both strata over-weight parse_fail/dissent
far above their natural rate. The manifest records the exact per-cell counts, and any short
cell is logged, never padded.
