# Gate 2 (10M) — mixture assembled

**Date:** 2026-09-23 · **Scale:** 11.4× the Gate-1 pilot · `data/sft/gate2_mixture.jsonl`

```
23,640 rows / ~9,899,900 tokens   decontam dropped: 0   study-only rows: 0   pools short: 0
```

## Target C generation

The one pool a hosted teacher had to produce. Ling-3.0-flash (Q6_K, ~101 GB, Vulkan + native MTP
speculative decoding) served on dashi, driven as an agent through the ADR-003 ClickHouse sandbox.

| run | requested | kept | failed | note |
|---|---|---|---|---|
| smoke | 5 | 5 | 0 | one rep of each new family, all passed |
| prompt-fix check | 1 | 1 | 0 | verified the system-prompt fix |
| main batch | 180 | 167 | 13 | `--reps 36 --run-offset 1` |
| top-up | 100 | 90 | 10 | `--run-offset 37`, self-sized from the file |
| final top-up | 30 | 29 | 1 | `--run-offset 57` |

Merged slice: **350 trajectories / 1,059,086 tokens**, 0 duplicate `(task, run_ix)`, all
oracle-passed, 7 families.

| family | kept | | family | kept |
|---|---|---|---|---|
| `mlc_churn_rare` | 63 | | `mlc_energy_load` | 41 |
| `mlc_credit_leak` | 63 | | `mlc_widget_defect` | 29 |
| `mlc_upsell_join` | 63 | | `mlc_delivery_time` | 29 |
| `mlc_ticket_route` | 62 | | | |

`widget_defect` and `delivery_time` carry over from the pilot and were not regenerated.
`energy_load` is short because of its yield — see below.

## Two defects found and fixed during generation

**A broken client idiom was being taught.** 55 of the 58 pilot trajectories contain a ClickHouse
connection error: the teacher calls `clickhouse_connect.get_client()` bare, fails, then burns three
or four of its 24 steps discovering the real connection details. That was already in the Gate-1
training data. The cause was the system prompt, which claimed the env "points at your scratch
database, so a client built from it reads/writes there by default" and then showed an example
referencing a `client` it never constructed. Measured on the same task after stating the idiom
explicitly: **3,506 → 1,572 tokens, 10 → 7 tool calls, ~105s → 58s, connection error yes → no**,
still clearing the bar. Every trajectory generated afterwards is clean (59/350 remaining are the
pre-existing pilot and smoke rows).

The token drop overstates the gain — error tracebacks are tool *output*, so they were padding
records cheaply. The real win is the recovered steps and removing a wrong first move from the
training signal.

**Top-ups would have silently duplicated data.** `generate()` always counted run indices from 0,
and the run index *is* the dataset seed, so extending the slice would have regenerated identical
datasets. Nothing downstream catches that: the assembler decontaminates against dsbench, not
against targetC itself. Added `--run-offset`.

## Open: `mlc_energy_load` yield

41 kept of 62 attempts (~66%), against 95–100% for every other family. Failures split between
near-misses (lift 20.9%, 23.3% against a 25% bar) and collapses (lift −91%, −185%, −200%).

I guessed this was lag-feature collapse — the test table deliberately has no `load_mw`, so lags are
unavailable across the horizon. **The kept trajectories refute that:** 35 of 36 passing runs use
shift/rolling/lag features. Whatever separates pass from fail there, it is not the presence of
lags, and the failed trajectories were discarded, so that is as far as the evidence goes.

Added `--fail-out` so the next run can diagnose it properly. Rejected trajectories are never
training data (`meta.verification.oracle_passed = false`) but are now inspectable.

## Proportion fidelity

Within 0.4pp of the pilot on every bucket, so the 11.4× scale is faithful:

| bucket | pilot | gate 2 | | bucket | pilot | gate 2 |
|---|---|---|---|---|---|---|
| targeted:sql | 5.7% | 5.8% | | general_code | 7.4% | 7.5% |
| targeted:ml | 10.8% | 10.4% | | swe | 12.4% | 12.4% |
| ds_notebooks | 21.9% | 21.9% | | replay | 28.6% | 28.8% |
| text_to_sql | 13.2% | 13.2% | | | | |

## Next

Tokenize with assistant-only masking (`build_masked_dataset.py`), then train. At the Gate-1 rate
(296 tok/s, rank 4 / seq 2048) a 9.9M-token epoch is roughly **4,900 steps ≈ 10 hours**.
