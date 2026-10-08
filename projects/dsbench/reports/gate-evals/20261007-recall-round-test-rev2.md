# The recall round's test: Revision 2's adapter against the base

ADR-004, item 12. Run 2026-10-07 on dashi, in one window of 1 hour 54 minutes, before Revision 2.1's
training (armed behind it). The test is `target_a_eval round-items`: 204 held-out items on the
round's cells.
- **The target cells (156 items):** ClickHouse weekend-filtered and workweek; DuckDB's
  weekday-numbering and three weekend families.
- **The guard cells (48 items):** PostgreSQL's and MySQL's weekend-filtered and workweek.

The test is paired on one server, base against adapter, as the Target A test is. The numbers are
in `20261007-recall-round-test-rev2.json`.

## Summary

- **Revision 2 already writes ClickHouse's new shapes right.**
  - The weekend beside a category filter goes from 1 of 24 to 23 of 24, and Monday to Friday from
    1 of 24 to 24 of 24 (McNemar p = 5e-7 and 2e-7).
  - Revision 2 trained neither shape: it learned the weekday numbers well enough to compose them.
  - The adapter's one miss is the old `IN (1, 7)`.
- **So dsbench's `da_weekend_delay` is not a shape the adapter lacks.** The adapter's `IN (1, 6)`
  there came in a multi-step agent run on the aviation warehouse. Held out, in one query, the same
  convention holds 47 times in 48.
  - The round's 78 ClickHouse rows can't show a gain on this test, which is at its ceiling. If they
    teach anything, dsbench's weekend problem will show it.
- **DuckDB is the real gap, and Revision 2 barely moves it.** It goes from 73 to 85 of 108, and no
  cell moves significantly:
  - weekday-numbering 25 → 28 of 36;
  - weekend-filtered 13 → 18 of 24;
  - weekend-flag 13 → 17 of 24;
  - workweek 22 → 22 of 24.
  
  The adapter's wrong DuckDB replies all use MySQL's numbers (`IN (1, 7)`, `BETWEEN 2 AND 6`,
  Sunday = 1), except one `IN (6, 7)`. That is the mistake the round's 102 DuckDB rows answer.
- **The guards hold: 47 → 46 of 48.**
  - MySQL: 24 of 24 in both states.
  - PostgreSQL: 23 → 22. The adapter lost two weekend-filtered items to `EXTRACT(DOW ...) IN (1,
    7)`, MySQL's weekend in PostgreSQL's Sunday-0 numbering, and gained one workweek item.
  - p = 1, but watch it in Revision 2.1.
- **The old belief still shows in the traces:** 87 of the adapter's 156 target traces state Sunday
  = 1 somewhere (the base: 121). In the Target A test it was 29 of 60.
- **Reasoning keeps its length:** median 1,664 tokens on the target cells (base 1,721). Replies cut
  at 12,288 tokens: 4 (base 6).

## What was run

- **The window** (`rev2_generate_window.sh` with `LORA` and `PARITY=1`; box clock 20:37 to 22:31,
  CEST 22:36 to 00:30): the battery's server with Revision 2's final adapter
  (`2026-10-06-qwen36-rev2-gate-final/lora.gguf`).
  - **Parity:** scale 0 gave the bare base's replies on 8 of 8 prompts, and again after scale 1.
    Scale 1 changed 4 of them.
  - **Timing:** base 55 minutes, adapter 57.
- **The checks** (the Mac's sandboxed engines): `target_a_hints verify`, then `target_a_eval
  recheck` on two more tables of each domain, then `compare`.
  - The recheck caught one of the base's replies, a lucky count, and none of the adapter's.
  - Production was off. OCR was stopped and restored.
- **The training followed:** Revision 2.1's window, armed at 20:38 once this window's server was
  up, started at 22:32 when this one had ended and OCR was back.

## Results

| Cell | n | Base | Adapter | Base only | Adapter only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| **Target** | 156 | 75 | 132 | 12 | 69 | 7e-11 |
| ClickHouse weekend-filtered | 24 | 1 | 23 | 0 | 22 | 5e-7 |
| ClickHouse workweek | 24 | 1 | 24 | 0 | 23 | 2e-7 |
| DuckDB weekday-numbering | 36 | 25 | 28 | 5 | 8 | 0.58 |
| DuckDB weekend-filtered | 24 | 13 | 18 | 2 | 7 | 0.18 |
| DuckDB weekend-flag | 24 | 13 | 17 | 3 | 7 | 0.34 |
| DuckDB workweek | 24 | 22 | 22 | 2 | 2 | 1 |
| **Guards** | 48 | 47 | 46 | 2 | 1 | 1 |
| PostgreSQL weekend-filtered | 12 | 12 | 10 | 2 | 0 | 0.5 |
| PostgreSQL workweek | 12 | 11 | 12 | 0 | 1 | 1 |
| MySQL weekend-filtered | 12 | 12 | 12 | 0 | 0 | 1 |
| MySQL workweek | 12 | 12 | 12 | 0 | 0 | 1 |

- **Statuses:**
  - base: 123 verified, 73 wrong, 2 errors, 6 unfinished;
  - adapter: 178 verified, 22 wrong, 4 unfinished.
- **DuckDB's workweek** is the one DuckDB cell the base already gets right (22 of 24). Its Monday
  to Friday is `BETWEEN 1 AND 5` in DuckDB's numbering, the same as in ISO. Under MySQL's belief it
  would be 2 to 6, which the base wrote once and the adapter twice. The base's other miss was a
  reply it never finished.

## Reading it

- **What Revision 2.1 can show on this test is DuckDB.** The ClickHouse cells are at 47 of 48 and
  the guards at 46 of 48, so neither has room to improve. The DuckDB cells are at 85 of 108.
- **The round's ClickHouse rows still have a job: the agent context.** dsbench's `da_weekend_delay`
  is the test for them. Revision 2 passed it once in five, writing `IN (1, 6)` three times.
- **The guards are the regression check.** DuckDB's rows teach that `dayofweek` counts Sunday 0.
  If MySQL's `DAYOFWEEK` starts to follow, its 24 guard items fall.
- **One more reading for Revision 2.1's run:** its base replies against this run's base. That pair
  is an A/A check across windows: the same items and seeds, a different day.
