# Target A test: Revision 2's adapter against the base, on prompts training never saw

ADR-004 Revision 2, action item 10. Run 2026-10-06 on dashi, in one GPU window of 2 hours 27
minutes. The code is `sftgen/target_a_eval.py`, with `reasoning_pilot generate --lora-scale` and
`patches/rev2_generate_window.sh`. The raw outputs are in
`~/benchlab/runs/2026-10-06-qwen36-rev2-target-a-test/`, and the numbers are in
`20261006-target-a-test.json`.

## Summary

- **The adapter has learned ClickHouse's weekday numbering.** On the 60 ClickHouse weekday and
  weekend items, the base gets 10 right and the adapter 58. 49 items are right for the adapter
  alone and 1 for the base alone (exact McNemar p = 9e-14).
  - Weekdays: 9 of 36 → 34 of 36.
  - Weekends: 1 of 24 → 24 of 24.

  These prompts are new: their tables, columns and row nouns appear in no training row.
- **Nothing else got worse.** On the 14 other cells, 147 of 168 → 156 of 168 (17 items for the
  adapter alone, 8 for the base alone; p = 0.11).
  - ISO did not spread to the other dialects: DuckDB's, PostgreSQL's and MySQL's weekday and
    weekend items went from 58 to 61 of 72.
  - The timezone family went from 42 to 48 of 48 (p = 0.03, the only group to move at 5%).
- **The old belief comes back as a doubt.** In 29 of its 60 ClickHouse traces, the adapter still
  states Sunday = 1 somewhere: a median of 1 sentence, against 7 in every one of the base's 60
  traces. All 29 settle on ISO and are right.
- **Reasoning keeps its length.** The median reasoning is 1,489 tokens on the target cells (base
  1,545) and 1,519 on the others (base 1,460). Replies cut at 12,288 tokens: 6, against the base's 9.
- **What is left:**
  - The adapter's two ClickHouse misses both count Thursday as day 5, which is Friday.
  - DuckDB numbers Sunday 0, and in a third of its weekday and weekend items both states use
    MySQL's numbering instead: 15 of 24 right for the base, 16 for the adapter. Target A trained
    DuckDB only on the base's own verified replies, which can't fix a mistake the base makes.
- **A single table can't tell a right count from a lucky one.** Each reply is now also checked on
  two more tables (`recheck`).
  - This caught two of the base's replies, which counted Monday for Sunday on tables that hold as
    many Mondays as Sundays. It caught none of the adapter's.
  - On the training data, the same check finds one wrong row in the mixture: 1 of its 250
    ClickHouse recall rows.

## What was run

- **Items:** 228 Target A prompts from `target_a_eval items` (seed 20261006, tables of 1,500 rows,
  `data/sft/rev2_target_a_test_manifest.json`).
  - **Six held-out domains** (`synth.held_out_domain_names`): library loans, ride trips, hotel stays,
    parcel scans, clinic visits, bike rentals. Their columns have the training domains' shapes.
  - **Why new domains, not just new tables.** The model sees a table's schema and the question,
    never its rows. So a new table in a training domain would ask a training prompt again: the
    training drew 124 of ClickHouse's 126 weekday prompts and all 18 of its weekend ones.
  - **Checked against the mixture:** none of the test's prompts is a training prompt, and none of
    its 4,326 rows names a held-out table or timestamp column.
  - **The question wordings are the training ones,** so this measures the conventions on unseen
    tables, not on unseen phrasing.
  - **Per cell:** ClickHouse weekday 36, ClickHouse weekend 24, and the 14 other cells 12 each.
- **Serving:** one server for both states. `battery_server.sh` loads the final adapter
  (`2026-10-06-qwen36-rev2-gate-final/lora.gguf`, the one the gate passed) with the global scale
  at 0, 8 slots, thinking on.
  - Every request named its scale: 0 for the base, 1 for the adapter. Each state was a step of its
    own, since requests at different scales never batch together.
  - Both states used the generation's sampling (temperature 0.6, top-p 0.95, top-k 20, min_p 0.05)
    and the same seed for an item, with a budget of 12,288 tokens.
- **Parity, first:** a server without the adapter, then scales 0, 1 and 0 on this one, greedy, on
  8 prompts. Scale 0 equalled the bare base on all 8, and again on all 8 after scale 1. Scale 1
  changed 4 of the 8: the adapter is applied, and stays close to the base on short replies.
- **The window, box clock:** 15:50 to 18:17.
  - Base: 70 minutes, 546,784 tokens at 129 a second.
  - Adapter: 74 minutes, 491,089 tokens at 110 a second.
  - No request failed. OCR was restored at the end. Production was off throughout, as the box's
    restart had left it.
- **The check, on the Mac's sandboxed engines:**
  1. `target_a_hints verify` runs each reply's SQL on its item's table, rebuilt from its seed after
     the gold SQL reproduces the truth there. A reply verifies when it finished, with reasoning,
     as one ```sql block whose count is the truth.
  2. `target_a_eval recheck` runs every verified reply's SQL, and the gold SQL, on two more tables
     of the item's domain. A reply is right only if the two agree there too.
  3. `target_a_eval compare` pairs the two states item by item.

## Results

"Right" is verified and rechecked; "base only" and "adapter only" are the items one state gets
right and the other doesn't; p is the exact McNemar test on those.

| Group | n | Base | Adapter | Base only | Adapter only | p |
|---|---:|---:|---:|---:|---:|---:|
| **Target: ClickHouse weekday and weekend** | 60 | 10 | 58 | 1 | 49 | 9.1e-14 |
| **The 14 other cells** | 168 | 147 | 156 | 8 | 17 | 0.11 |
| … weekday numbering | 36 | 30 | 31 | 3 | 4 | 1 |
| … weekend flag | 36 | 28 | 30 | 4 | 6 | 0.75 |
| … timezone direction | 48 | 42 | 48 | 0 | 6 | 0.031 |
| … month bucket | 48 | 47 | 47 | 1 | 1 | 1 |
| … ClickHouse | 24 | 21 | 24 | 0 | 3 | 0.25 |
| … DuckDB | 48 | 38 | 39 | 6 | 7 | 1 |
| … PostgreSQL | 48 | 41 | 45 | 2 | 6 | 0.29 |
| … MySQL | 48 | 47 | 48 | 0 | 1 | 1 |

Per cell:

| Cell | n | Base | Adapter | Base only | Adapter only | p |
|---|---:|---:|---:|---:|---:|---:|
| ClickHouse weekday | 36 | 9 | 34 | 1 | 26 | 4.2e-07 |
| ClickHouse weekend | 24 | 1 | 24 | 0 | 23 | 2.4e-07 |
| ClickHouse timezone | 12 | 9 | 12 | 0 | 3 | 0.25 |
| ClickHouse month | 12 | 12 | 12 | 0 | 0 | 1 |
| DuckDB weekday | 12 | 8 | 8 | 3 | 3 | 1 |
| DuckDB weekend | 12 | 7 | 8 | 2 | 3 | 1 |
| DuckDB timezone | 12 | 12 | 12 | 0 | 0 | 1 |
| DuckDB month | 12 | 11 | 11 | 1 | 1 | 1 |
| PostgreSQL weekday | 12 | 10 | 11 | 0 | 1 | 1 |
| PostgreSQL weekend | 12 | 10 | 10 | 2 | 2 | 1 |
| PostgreSQL timezone | 12 | 9 | 12 | 0 | 3 | 0.25 |
| PostgreSQL month | 12 | 12 | 12 | 0 | 0 | 1 |
| MySQL weekday | 12 | 12 | 12 | 0 | 0 | 1 |
| MySQL weekend | 12 | 11 | 12 | 0 | 1 | 1 |
| MySQL timezone | 12 | 12 | 12 | 0 | 0 | 1 |
| MySQL month | 12 | 12 | 12 | 0 | 0 | 1 |

Statuses over the 228 items:
- **Base:** 159 verified (2 of them coincidences, below), 57 wrong, 9 cut at the budget, 2 engine
  errors, 1 with prose around its SQL.
- **Adapter:** 214 verified (all held on the recheck), 7 wrong, 6 cut, 1 engine error.

### ClickHouse's weekdays

- **The base answers with MySQL's numbering.** It writes `dayOfWeek(ts) IN (1, 7)` for the
  weekend, Thursday as 5 and Saturday as 7. Every one of its 60 target traces states Sunday = 1, a
  median of 7 times.
  - Its 9 right weekday answers write ISO's numbers: in those traces it settled on ISO anyway,
    after stating Sunday = 1 along the way.
  - It got 1 of the 24 weekends.
- **The adapter answers with ISO.** 34 of 36 weekdays, every day of the week complete but Thursday
  (7 of 9), and 24 of 24 weekends. Its two misses both write `toDayOfWeek(...) = 5` for Thursday,
  MySQL's number.
  - One of the mixture's 250 recall rows makes that same mistake (the training audit, below).
    That one row can't be said to have caused them: the base makes the same mistake on its own.
- **It still doubts, less.**
  - 29 of the adapter's 60 traces state Sunday = 1 somewhere, typically once, as a question it then
    answers: "Wait, is there a chance `toDayOfWeek` returns 1=Sunday?" All 29 are right.
  - The training dropped the recall traces that doubted (decision 2), so the rows held no such
    doubt. This is the base's belief resurfacing.
  - One trace says "Sunday = 1 means ISO" and still writes ISO's numbers.

### The other cells

- **Timezones improved.** The base's six misses there:
  - PostgreSQL's three converted in the wrong direction: `ts - INTERVAL '4 hours'` for a city at
    UTC-4. That is the trap the timezone family teaches.
  - ClickHouse's two were engine errors: a `CASE` of `ts + INTERVAL n HOUR` that ClickHouse
    rejects.
  - One reply was cut at the budget.

  The adapter got all 48. Its timezone rows were the base's own verified replies, so rejection
  sampling reinforced what the base already did right some of the time.
- **DuckDB's weekday function is the gap that is left.** DuckDB's `dayofweek` numbers Sunday 0.
  Both states use MySQL's numbering for it in about a third of the items: `dayofweek(ts) IN (1, 7)`,
  or Saturday as 7. Base 15 and adapter 16 of 24. The prefill route that fixed ClickHouse would
  apply, but nothing in Revision 2 used it for DuckDB.
- **MySQL, where Sunday = 1 is right, is unchanged.** 23 of 24 → 24 of 24 on weekdays and
  weekends. The adapter learned whose numbering Sunday = 1 is, not to avoid it.

## The recheck, and what it found in the training data

- **Why.** Every Target A item asks for a count. With 1,500 rows spread over a year, two weekdays'
  counts sometimes tie on a table. The base counted Monday (day 1) for Sunday on two items whose
  tables held 215 Mondays and 215 Sundays, and 210 and 210. Both verified.
  `recheck` runs a verified reply's SQL and the gold SQL on two more tables of the same domain
  (seeds a prime stride past the item's). There, both of these failed, and the other 157 of the
  base's verified replies and all 214 of the adapter's held.
- **The generation's own check has the same blind spot,** so the training's Target A rows were
  rechecked too:

  | Pool | Verified | Held | In the mixture and wrong |
  |---|---:|---:|---:|
  | The 14 other cells (plain) | 222 | 222 | 0 |
  | ClickHouse recall | 490 | 487 | 1 |
  | ClickHouse recall, the top-up | 233 | 232 | 0 |

  Of the four coincidences, three doubted the convention, so the assembler had dropped them. The
  fourth is in the mixture, 1 of its 250 recall rows: Thursday as `toDayOfWeek(...) = 5`. A later
  generation's Target A replies should go through the recheck before assembly.

## What this does not show

- **Unseen phrasing:** the questions are worded as in training. The held-out probe and dsbench
  (k = 5), on aviation data and through pi, are the far transfer, and the next step.
- **Sampling noise per cell:** each item was answered once per state. The target result is far
  outside noise, and the other cells' totals are within it (p = 0.11). A cell of 12 can only show
  a collapse.
