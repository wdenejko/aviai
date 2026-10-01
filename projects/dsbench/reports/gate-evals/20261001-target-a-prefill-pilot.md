# Target A prefill pilot: the convention in the base's own reasoning

ADR-004 Revision 2, action item 2, and the two routes decision 2 proposes. Run 2026-10-01 on dashi,
in one GPU window. The code is `sftgen/prefill.py` and `sftgen/target_a_hints.py`. The raw outputs
are in `~/benchlab/runs/2026-10-01-qwen36-target-a-prefill-pilot/`, and the numbers are in
`20261001-target-a-prefill-pilot.json`.

## Summary

- **A `recall` prefill supplies Target A's ClickHouse rows.** The base's own plain trace is cut
  where it first turns to the weekday function. The convention is written there, and the base
  continues. The result:
  - 95 of 96 replies verify;
  - none credits the prompt or names a hint;
  - every row has kept replies;
  - the traces are the base's full length: a median of 1,541 and 1,561 reasoning tokens for
    weekdays and weekends, against 1,504 and 1,446 for its plain replies.
- **Opening the thinking block with the convention (`start`) keeps almost as many, 92 of 96, but
  shortens the trace** to two-thirds of the base's length: a median of 1,026 and 961 tokens.
- **Plain sampling can't supply the rows.** 7 of 96 plain replies verify. 18 of the 24 rows were
  never right in 4 tries, and every plain trace states Sunday = 1 along the way. 250 rows would
  take about 19 hours of box time.
- **The old belief comes back as a doubt.** About half of the kept prefilled traces raise Sunday
  = 1 again after the prefill before settling on ISO, for example "is there any chance
  `toDayOfWeek` returns 1 for Sunday". Keeping only the `recall` traces that don't still leaves a
  kept reply on 23 of 24 rows, at the base's length (a median of 1,475 tokens).
- **For decision 2:** Target A's ClickHouse weekday and weekend rows come from the `recall`
  prefill, and no teacher is needed. Whether to keep the traces that doubt is the open question;
  250 rows take about 1 hour of box time with them and 2 without.

## What was run

- **Rows:** 24 ClickHouse rows from `dialect_conventions.py`, 12 weekday and 12 weekend (6 domains
  x 2 tables, seed 20261001, tables of 1,500 rows), with the reworded questions. None overlaps
  the battery.
- **Serving:** the base (APEX I-Mini, no LoRA), thinking on, with Qwen's sampling (temperature
  0.6, top-p 0.95, top-k 20), 8 slots, at most 16,384 tokens a reply.
  - Every reply went through the raw completion path: the server's template (`/apply-template`),
    then `/completion`, with `<think>` and `</think>` preserved. Plain replies had an empty
    prefill.
  - Checked locally before the window: with nothing prefilled, this path reproduces the chat
    endpoint's reply token for token.
- **The sentence**, the same in every prefilled reply, with each function it names checked on its
  engine: "In ClickHouse, `toDayOfWeek(date)` (alias `dayOfWeek`) returns 1 for Monday, 2 for
  Tuesday, ..., 7 for Sunday: the ISO numbering. (MySQL's `DAYOFWEEK` is the one that starts at
  Sunday = 1.)"
- **Phase 1** (48.5 minutes, 360,234 tokens, 124 tokens/s): each row answered 4 times plain and 4
  times `start`.
- **Phase 2** (17.1 minutes, 139,975 tokens, 137 tokens/s): each plain reply cut and continued
  once (`recall`). All 96 had a cut.
- **The window:** 66 minutes, smoke test included. The thermal governor never stepped in. OCR was
  stopped once loaded and restored at the end.
- **Checks:** every reply went through `target_a_hints verify` on the sandboxed ClickHouse. A
  reply counts when it finished with reasoning, is one sql block and nothing else, and returns the
  row's truth. A prefilled reply is kept only if what it wrote after the prefill doesn't credit
  the prompt with the convention or name a hint.

## Results

Kept replies of 48 per cell, with the rows (of 12) that have at least one, and the median
reasoning tokens of all replies:

| Arm | Weekday | Weekend | Rows | Median tokens (weekday / weekend) |
|---|---:|---:|---:|---:|
| plain | 5 | 2 | 4 + 2 | 1,504 / 1,446 |
| `start` | 45 | 47 | 12 + 12 | 1,026 / 961 |
| `recall` | 47 | 48 | 12 + 12 | 1,541 / 1,561 |

No reply in any arm cites anything. Every kept prefilled reply fits 4,096 tokens with its prompt;
one of the 7 plain ones doesn't.

### The base alone

This is the plain-sampling route.
- 7 of 96 replies verify. Of the 12 weekday rows, 8 were never right in 4 tries, 3 once and 1
  twice. Of the 12 weekend rows, 10 were never right, and 2 once.
- Every plain trace raises Sunday = 1 at some point, counted as below, including the 7 that end
  right.
- All 42 weekend misses are `IN (1, 7)`. The weekday misses fall on the numbers that a
  Sunday-first count gives.
- 6 replies ran to the 16,384-token limit, so a plain reply averages 2,562 tokens.
- At this yield, 250 rows would take about 3,400 replies: 8.8M tokens, about 19 hours of box time.
  And the rows they'd cover are the ones the base happens to get right.

### `start`

- 92 of 96 kept.
  - Of the 3 wrong, the old belief won twice ("ClickHouse uses Sun=1"). The third is an
    arithmetic slip: Wednesday = 4 under Monday = 1.
  - One reply looped to the token limit after writing its answer.
- The traces run two-thirds of the base's length. Given the answer before it starts, the base
  deliberates less. That is the brevity the hint pilot's survivors showed (557 tokens), milder.
- Every row begins with the same sentence.

### `recall`

- **The cut:** a median of 367 characters into the plain trace (about 109 tokens), at most 666
  (about 195 tokens). Every plain reply had one.
- **95 of 96 kept.** The miss is an arithmetic slip under the right numbering: the base lists
  "4 for Thursday", then writes "So Thursday is 5".
- **The traces are the base's own length,** because everything in them but the one sentence is
  the base's own writing.
- **The splice reads as the base's own,** for example:

  > 3.  **Determine ClickHouse SQL Syntax for Day of Week**:
  >    - [the sentence]
  >    - Let's verify: ClickHouse `toDayOfWeek` documentation says: Returns the number of the
  >      day of the week (1-7, where 1 is Monday, 7 is Sunday).
  >    - So, Sunday = 7.

  No continuation calls the sentence "the statement above", and none says "as I said" about it.
  87 of the 95 kept continuations open by checking it ("Wait, let's verify ...").

### The old belief comes back

After the prefill, 40 of the 92 kept `start` traces, and 49 of the 95 kept `recall` traces, state
Sunday = 1 for ClickHouse again, then settle on ISO.

- **What they look like.** Most are doubts: "Wait, is there any chance `toDayOfWeek` returns 1 for
  Sunday", or "Some sources say `toDayOfWeek` returns 1 for Sunday in older versions". Some are
  misremembered documentation: "dayOfWeek(date) returns ... (1-7, Sunday is 1)". A few contrast
  ClickHouse with other dialects, which is right. One briefly flips ("I just remembered that in
  ClickHouse, `toDayOfWeek` actually returns 1 for Sunday") and flips back.
- **What a filter would cost.** These traces would teach the model to doubt the convention and
  then override the doubt. Dropping them leaves:
  - 46 `recall` traces, with a kept reply on 23 of 24 rows;
  - a median of 1,475 tokens, so the base's length holds. The doubting traces run longer, at
    1,634.

  For `start`, the clean traces shrink to a median of 888 tokens.
- **How it was counted:** a regex for sentences after the prefill that state Sunday = 1 and don't
  name MySQL, with two samples of 15 matches read by hand. It also counts the contrasts, so it
  slightly overstates the doubts.

## What it means for the retrain

1. **Decision 2: the `recall` prefill, with no teacher.**
   - It is on-policy except for one checked sentence.
   - It keeps the base's length.
   - Its rows are licence-clean, like any of the base's own replies.
2. **Open question for the owner: keep the traces that doubt, or not.**
   - Keeping them: 250 rows take about 250 replies, about 1 hour of box time.
   - Dropping them: about 520 replies and 2 hours.
   - Proposed: drop them. The cleaner supervision costs an hour.
3. **Before scale:**
   - Paraphrase the sentence into a few checked variants, so that the rows don't all carry one
     sentence.
   - Stop the plain phase at 512 tokens, since every cut fell within about 200. A `recall` reply
     then costs about 2,000 tokens.
   - Move the doubt filter from this analysis into `target_a_hints verify`, if the owner keeps
     it.
4. **The other cells don't need a prefill.** Postgres, MySQL and DuckDB, and the month and
   timezone families, come from the base's own verified plain replies: 146 of the hint pilot's
   192 verified.

**Limits:** 24 rows and one sentence. Each plain reply was cut and continued once. The doubt count
comes from a regex.
