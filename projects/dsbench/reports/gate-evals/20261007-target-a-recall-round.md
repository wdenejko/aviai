# Target A's recall round: ClickHouse's weekend shapes and DuckDB's weekday numbering

ADR-004 Revision 2, action item 11. Run 2026-10-07 on dashi in two generation windows: about two
hours, then 14 minutes.
- **The code:** `target_a_hints recall-round-items`, `prefill splice` and `recut`, `reasoning_pilot
  generate` in `patches/rev2_generate_window.sh`.
- **The checks:** `target_a_hints verify` and `target_a_eval recheck`, on the Mac's sandboxed
  engines.
- **Where things are:** the raw outputs in `~/benchlab/runs/2026-10-07-qwen36-ta-recall-round/`;
  the numbers in `20261007-target-a-recall-round.json`.

## Summary

- **180 rows train, from 468 items.** That is about the 185 the round was sized for.
  - ClickHouse gives 78 (expected about 85): 45 weekend-filtered, 14 weekend-flag, 19 workweek.
  - DuckDB gives 102 (expected about 100): 53 weekday-numbering, 15 weekend-filtered, 15
    weekend-flag, 19 workweek.
- **A row trains when three things hold:** its reply verified, it held on two more tables, and it
  states no Sunday = 1 after the convention.
  - **Verification:** 429 of the 468 replies verified.
  - **The recheck:** it failed none of the 429.
  - **Citing:** no reply cites the convention as if told.
- **Doubt is the main loss, as in Revision 2.** 249 of the 429 verified replies (58%) state Sunday
  = 1 again before settling on the right numbering. Decision 2 drops them. Revision 2's top-up,
  with the same ClickHouse wordings, lost 57% this way.
- **Every wrong reply is the old belief overruling the convention.** 36 replies are wrong:
  - in ClickHouse, `IN (1, 7)`: MySQL's weekend;
  - in DuckDB, MySQL's day numbers, with two hybrids `IN (1, 6)` and one `IN (6, 7)` (ClickHouse's
    numbering carried to DuckDB);
  - three more replies called functions that don't exist.
- **DuckDB's belief resists the convention more.** Plain, the base states Sunday = 1 in 214 of 216
  ClickHouse traces and in 155 of 252 DuckDB ones.
  - After the convention, it overrules it in 3 of 216 ClickHouse replies and in 33 of 252 DuckDB
    ones.
  - A likely reason is the name: DuckDB's `dayofweek` is spelled as MySQL's `DAYOFWEEK`, which
    does number Sunday 1.
- **A false cut, fixed during the run** (PR #49):
  - The cut took the prose "weekday (Monday to Friday)" for MySQL's `WEEKDAY(`. 71 of the 96
    workweek items were cut at their first sentence, so the convention opened the trace.
  - The 71 were re-cut and answered again in the second window. Their first replies are not used.
  - **On the same 71 prompts, the two placements differ the way the prefill pilot found.** The
    convention first: 44 rows train, at a median of 1,012 reasoning tokens. The recall cut: 28 rows
    at 1,470.
- **With the round, the mixture grows 3.6%:**
  - Target A goes from 466 rows (0.80M tokens) to 665 (1.16M), once its budget is opened.
  - Every other pool keeps exactly Revision 2's rows.
  - The total is 4,525 rows and 10.17M tokens.

## What was run

- **The items:** 468 plain items on the training domains (`data/sft/target_a_recall_round_manifest.json`,
  PR #48). Each item carries its dialect's convention sentence: ClickHouse wordings 0, 1 and 3, and
  DuckDB's three.
- **Window 1** (box clock 11:58 to 14:01; CEST about 13:56 to 15:59): the bare base, as in Revision
  2's generation. Production was off; OCR was stopped and restored.
  - **`rr_plain`:** 468 replies of 512 tokens, in 27 minutes. All of them ran to the budget, as the
    plain phase means them to.
  - **The splice:** 468 of 468 found their cut.
  - **`rr_recall`:** 468 continuations in 94 minutes, all finished.
- **The false cut** was found from `rr_plain` while `rr_recall` ran, and fixed on the Mac (PR #49).
  `prefill recut` split the splice into two sets:
  - 397 items whose prefill stays, whose replies stand;
  - 71 re-cut workweek items: 41 ClickHouse, 30 DuckDB.
- **Window 2** was armed behind the first. ARM=1 waits while any llama-server but OCR's runs. It
  started at 14:02, once OCR was back, and answered the 71 (`rr_recall_ww`) in 13 minutes.
- **The checks:** each half was verified against its own items, so the 71 first replies drop out
  (`patches/README.md`, "The recall round"). Then the recheck ran on two more tables of each
  domain.

## Results

| Dialect / family | Items | Verified | Wrong | Error | Doubting | Trains | Kept reasoning (median tokens) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ClickHouse weekend-filtered | 120 | 120 | 0 | 0 | 75 | 45 | 1,500 |
| ClickHouse weekend-flag | 36 | 33 | 3 | 0 | 19 | 14 | 1,542 |
| ClickHouse workweek | 60 | 59 | 0 | 1 | 40 | 19 | 1,147 |
| DuckDB weekday-numbering | 120 | 104 | 16 | 0 | 51 | 53 | 1,720 |
| DuckDB weekend-filtered | 60 | 45 | 14 | 1 | 30 | 15 | 2,046 |
| DuckDB weekend-flag | 36 | 32 | 3 | 1 | 17 | 15 | 2,027 |
| DuckDB workweek | 36 | 36 | 0 | 0 | 17 | 19 | 1,696 |
| **All** | **468** | **429** | **36** | **3** | **249** | **180** | |

"Doubting" counts verified replies that state Sunday = 1 again after the convention. "Error" counts
SQL the engine refused: the invented `isoweekend()` and `toISOWeekday()`, and `extract('dow', ...)`.

**The base's belief, before any convention.** Each plain trace was searched over its 512 tokens for
a statement of Sunday = 1:

| Dialect | Plain traces | State Sunday = 1 | Before the cut | Overrule the convention after it |
|---|---:|---:|---:|---:|
| ClickHouse | 216 | 214 | 0 | 3 (1.4%) |
| DuckDB | 252 | 155 | 0 | 33 (13%) |

No statement comes before its trace's cut, so the splice removes every one.

**The wordings.** The six convention sentences say the same thing, and two of them stand out in
every family:
- **DuckDB wording 0 is overruled most:** 18 wrong of 89, against 6 of 86 for wording 1 (Fisher p =
  0.015). Wording 1 ends "Sunday = 1 is MySQL's `DAYOFWEEK`, not DuckDB's".
- **ClickHouse wording 1** ("Let me recall ClickHouse's numbering: ...") **draws the most doubt:** 43
  of its 56 verified replies (77%), against 45 of 84 for wording 0 (p = 0.007). Revision 2's
  top-up found the same: wording 1 doubted most there too (68%).
- The DuckDB comparison is post hoc, among six wordings; the ClickHouse one repeats. They matter
  only if more of this data is generated: then use DuckDB's wordings 1 and 2, and ClickHouse's 0
  and 3.

## The false cut, and what it showed

- **The cause:** `prefill.cut` ends a plain trace where the base first turns to the weekday
  function. Its pattern for MySQL's `WEEKDAY(` allowed a space before the parenthesis. The workweek
  prompts make the base restate "on a weekday (Monday to Friday)" in its first sentence, and the
  pattern matched that.
- **The effect:** in 71 of the 96 workweek traces, the cut fell before the trace had turned to any
  function: at character 0, or just after "1. **Understand User Goal**:". The convention then
  opened the trace. That is the `start` placement, which decision 2 turned down for its shorter
  traces.
- **The fix** counts `WEEKDAY(` only as code. Re-cut from the same plain replies:
  - the 71 fall on `toDayOfWeek`, `dayOfWeek` or `dow`;
  - the median cut moves from character 58 to 440 (ClickHouse) and from 0 to 156 (DuckDB);
  - none of the other 397 items moves.
- **Revision 2 is not affected:** its 768 recall cuts all fell on `dayOfWeek` or `toDayOfWeek`.

The 71 prompts were answered both ways, so they compare the two placements directly:

| On the same 71 prompts | Verified | Doubting | Trains | Reasoning (median, all / kept) |
|---|---:|---:|---:|---:|
| The convention first (the false cut) | 69 | 25 | 44 | 1,120 / 1,012 |
| The recall cut | 70 | 42 | 28 | 1,627 / 1,470 |

- **Paired by prompt:**
  - 30 train only with the convention first, and 14 only with the recall cut;
  - 14 train both ways, and 13 neither.
- **This repeats the prefill pilot:** the convention first gives more rows, with about a third less
  reasoning. Given the fact before it starts, the base doubts less and reasons less.
- **The 44 short rows are not used.** Gate 2 failed partly on traces that taught brevity, and
  decision 2 chose recall to keep the base's length.

## The mixture with the round (a preview)

- **Revision 2 reproduces:** `assemble_rev2` with Revision 2's inputs and arguments rebuilt its
  mixture bit for bit (sha256 `e92a5b1781a3d909...`, as in its manifest).
- **The round's rows don't fit the current budget:**
  - The round adds two `--verified` pairs, whose ids don't overlap: `rr_recall.same` and
    `rr_recall_ww`.
  - Target A has a bucket of its own, 800,000 tokens. Revision 2 filled it with 466 of its 485
    eligible rows.
  - Kept at that budget, the round's 180 rows would push out, at random, about a third of Revision
    2's Target A rows, those that gave its gains.
- **With Target A's budget opened** (patched for the preview, not yet a code change):

| | Revision 2 | With the round |
|---|---:|---:|
| Target A rows | 466 | 665 |
| Target A tokens | 802,188 | 1,155,368 |
| Mixture rows | 4,326 | 4,525 |
| Mixture tokens | 9,821,199 | 10,174,379 |

- **What changes:**
  - The 665 are all 485 eligible Revision 2 rows plus the round's 180 (325,198 tokens). The 19 rows
    the old budget left out come in too: 13 ClickHouse recall rows and 6 from other cells.
  - No Revision 2 row drops out.
  - Every other pool keeps exactly Revision 2's rows: each pool draws from its own seeded
    generator.
  - Decontamination: no hits, and no row runs over the 8,192-token block.
- **The cost:** at Revision 2's 24.3 seconds a step, about 1,250 blocks would train in about 8.5
  hours.

## Reading it

- **The round makes the data it was built for.** 78 ClickHouse rows cover the two weekend shapes
  the adapter got wrong on dsbench's `da_weekend_delay`. 102 DuckDB rows cover the numbering both
  states got wrong in the Target A test.
- **The data is clean by construction:**
  - every wrong reply drops at verification;
  - every doubting one is dropped by decision 2;
  - the recheck found no lucky count this time.
- **What the rows can't show is what they teach.** That needs a retrain and the tests:
  - the Target A test (its DuckDB cells);
  - dsbench's `da_weekend_delay` and the probe;
  - the mini-battery gate, which shows that nothing else moved.
- **The new shapes have no held-out test yet.** The Target A test asks weekend-flag and
  weekday-numbering, not weekend-filtered or workweek. A test of the two shapes would need items on
  the held-out domains (`target_a_eval items`, extended).
