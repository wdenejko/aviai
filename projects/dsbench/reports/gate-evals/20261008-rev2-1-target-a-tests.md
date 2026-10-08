# Revision 2.1's Target A tests: the recall round's test and the Target A test

ADR-004, item 12. Run 2026-10-08 on dashi, in a window right after Revision 2.1's gate. A reboot
and the owner's use of the box split it into three server runs. Each test is paired on one server:
the base against Revision 2.1's final adapter, as Revision 2's tests were.
Revision 2's runs of the same items give two more readings:
- **Revision 2 against Revision 2.1:** the round's effect, across windows.
- **The base against itself across windows (A/A):** the same items and seeds on another day. It
  measures how far a state moves between windows by sampling alone.

The numbers are in `20261008-rev2-1-target-a-tests.json`.

## Summary

- **Revision 2.1 has learned DuckDB's `dayofweek`.** It is right in 38 of the 42 replies that call
  the function. Revision 2 was right in 0 of 18, and the base in 1 of 29 (Fisher p = 8e-12 and
  2e-14).
  - The function is spelled as MySQL's `DAYOFWEEK`, which numbers Sunday 1. DuckDB numbers Sunday
    0.
  - Revision 2 trained none of these replies: its 21 DuckDB weekday rows all used `EXTRACT(DOW
    ...)`, since the base's `dayofweek` replies failed verification.
  - The round's 102 DuckDB rows use `dayofweek` in 43.
  - Over the two tests, Revision 2.1 is right in 44 of its 51 `dayofweek` replies, and Revision 2
    in 0 of 23.
- **On the round's test, DuckDB goes from 76 to 95 of 108 against this window's base** (McNemar p
  = 0.003). Revision 2 reached 85.
  - Wrong DuckDB answers fall from 19 for Revision 2 to 6. Five of the six still use MySQL's
    numbers.
  - The other 7 misses don't answer: 5 replies ran to the budget, and 2 queries DuckDB rejected.
  - Against Revision 2, across windows, 85 → 95 is p = 0.10. The function is where the round's
    effect shows clearly.
- **ClickHouse stays at its ceiling:** 46 of 48 on the round's shapes (Revision 2: 47). Both
  misses are on Monday to Friday.
- **The guards hold.**
  - MySQL: 24 of 24 in every run. Every Revision 2.1 reply there uses `DAYOFWEEK` with Sunday = 1,
    so DuckDB's Sunday 0 did not carry over to MySQL's function of the same name.
  - PostgreSQL: 23 of 24. The one miss is `EXTRACT(DOW ...) IN (1, 7)`, which Revision 2 wrote
    twice. Counting the Target A test's PostgreSQL weekday and weekend items too, the two adapters
    wrote MySQL's numbers there 3 times in 96, and the two base runs 2 times in 96. That is noise,
    not a regression.
- **The old belief is stated less.** On the round's test, Revision 2.1 states Sunday = 1 in 58 of
  its 156 target traces, against Revision 2's 87 (p = 0.0009) and the base's 133.
  - ClickHouse: 21 → 11 of 48.
  - DuckDB: 66 → 47 of 108.
- **On the Target A test, Revision 2.1 keeps what Revision 2 had.**
  - ClickHouse's weekday and weekend: 12 → 57 of 60 against this window's base (p = 7e-13).
    Revision 2 had 58.
  - The 14 other cells: 155 → 156 of 168 (Revision 2: 156).
  - DuckDB's weekday and weekend cells: 15 → 19 of 24 (Revision 2: 16; p = 0.22).
- **Loops are now a large share of what is left.** 13 of Revision 2.1's 31 misses over the two
  tests are replies that ran to the 12,288-token budget, 10 of them writing a finished query out
  again and again.
  - The runs cut about as often as each other: base 1 15, Revision 2 10, base 2 9, Revision 2.1
    13, of 432 items each.
  - The base loops like this on its own: the mini-battery's calibration found it.
- **Reasoning keeps its length:** median 1,720 tokens on the round's target cells (base 1,682), and
  1,475 on the Target A test's (base 1,486).
- **The A/A puts a scale on the cross-window readings.** The base's own target score moved from 75
  to 79 of 156 between the two windows, with 36 items changing sides (p = 0.62). Small
  differences between Revision 2's window and Revision 2.1's are within that.

## What was run

- **The window** (`rev2_generate_window.sh` with `LORA` and `PARITY=1`;
  `~/benchlab/runs/2026-10-08-qwen36-rev2-1-tests/`):
  - **Arming:** armed at 08:28 behind the gate (box clock, about 1 h 59 min behind CEST), and
    started at 12:35 when the gate's window had ended.
  - **The server:** the battery's server with Revision 2.1's final adapter
    (`2026-10-08-qwen36-rev2-1-gate-final/lora.gguf`, the one the gate passed), global scale 0.
    Every request named its scale: 0 for the base, 1 for the adapter.
  - **The plan:** four steps, `rr_test:base rr_test:adapter ta_test:base ta_test:adapter`.
  - **Sampling:** the generation's (temperature 0.6, top-p 0.95, top-k 20, min_p 0.05), the same
    seed for an item in every run, and a budget of 12,288 tokens.
- **Parity, at each launch:** scale 0 gave the bare base's greedy replies on 8 of 8 prompts, and
  again after scale 1. Scale 1 changed 2 of the 8, all three times. Revision 2's adapter changed 4;
  on short replies both stay close to the base.
- **Timing (box clock):**
  - The round's test: base 12:38 to 13:25 (47 minutes, 386,362 tokens at 136 a second), adapter
    13:25 to 14:25 (60 minutes, 399,792 tokens at 111 a second).
  - The Target A test: base 512,402 tokens over three server runs (below), adapter 20:37 to 21:57
    (80 minutes, 513,444 tokens at 107 a second).
  - No request failed. Production was off. OCR was stopped at each launch and restored at each end.
    The first launch had to kill an OCR server that outlived the stop: the gate's window had
    restored it minutes before.
- **The Target A test's base state took three server runs.** Every step skips the items
  already answered, so the base's 228 replies come from three servers. The same model and
  settings served all three, and parity gave the same result each time.
  - **15:03, a reboot.** The box rebooted without a shutdown at 109 replies. No kernel error was
    logged, and the Mac lost its network at the same moment: most likely a power cut. All 109
    replies were intact. With the owner's agreement, the window was relaunched at 15:33.
  - **15:47, the owner needed the box.** The window was stopped at 146 replies. The owner's model
    then had to be rebuilt first: an unrelated model-file problem, fixed in
    `~/benchlab/runs/2026-10-08-flashnext-a1perm-rebuild/`.
  - **20:13, relaunched** once the owner said the box was free. The base's last 82 replies took
    22 minutes. The adapter's 228 then ran in one go.
- **The checks** (the Mac's sandboxed engines), as for Revision 2:
  1. `target_a_hints verify` runs each reply's SQL on its item's table.
  2. `target_a_eval recheck` runs every verified reply's SQL, and the gold SQL, on two more tables
     of its domain.
  3. A reply is right when it verified and held.

  The recheck caught one of the base's replies on the round's test, and none of Revision 2.1's on
  either test.
- **The items are held out from Revision 2.1's mixture too.** No prompt of either test is a
  training prompt. None of the mixture's 4,525 rows names a held-out table or timestamp column. The
  round's test was checked when it was built (PR #51), and the Target A test again for this run.

## The round's test

204 items: the round's target cells (156) and PostgreSQL's and MySQL's guard cells (48). "Base 1"
and "Revision 2" are the window of 2026-10-07; "base 2" and "Revision 2.1" are this one. The p
values are exact McNemar tests: base 2 against Revision 2.1, the test as designed, and Revision 2
against Revision 2.1, across windows.

| Cell | n | Base 1 | Revision 2 | Base 2 | Revision 2.1 | p, base 2 vs 2.1 | p, 2 vs 2.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Target** | 156 | 75 | 132 | 79 | **141** | 1e-12 | 0.16 |
| ClickHouse weekend-filtered | 24 | 1 | 23 | 2 | 24 | 5e-7 | 1 |
| ClickHouse workweek | 24 | 1 | 24 | 1 | 22 | 1e-6 | 0.5 |
| DuckDB weekday-numbering | 36 | 25 | 28 | 27 | 31 | 0.42 | 0.58 |
| DuckDB weekend-filtered | 24 | 13 | 18 | 15 | 21 | 0.07 | 0.45 |
| DuckDB weekend-flag | 24 | 13 | 17 | 13 | 20 | 0.09 | 0.45 |
| DuckDB workweek | 24 | 22 | 22 | 21 | 23 | 0.63 | 1 |
| … ClickHouse | 48 | 2 | 47 | 3 | 46 | 2e-13 | 1 |
| … DuckDB | 108 | 73 | 85 | 76 | **95** | 0.003 | 0.10 |
| **Guards** | 48 | 47 | 46 | 48 | 47 | 1 | 1 |
| PostgreSQL weekend-filtered | 12 | 12 | 10 | 12 | 11 | 1 | 1 |
| PostgreSQL workweek | 12 | 11 | 12 | 12 | 12 | 1 | 1 |
| MySQL weekend-filtered | 12 | 12 | 12 | 12 | 12 | 1 | 1 |
| MySQL workweek | 12 | 12 | 12 | 12 | 12 | 1 | 1 |

- **Statuses:**
  - Revision 2.1: 188 verified (all held on the recheck), 9 wrong, 2 engine errors, 5 unfinished.
  - Base 2: 128 verified (1 failed the recheck), 73 wrong, 3 unfinished.
- **A/A, base 1 against base 2:** target 75 → 79 (16 items for base 1 alone, 20 for base 2 alone),
  guards 47 → 48. No cell moves by more than 2.

### DuckDB, by the function a reply calls

Right of the replies in DuckDB's 108 target items that call each function:

| Function | Base 1 | Revision 2 | Base 2 | Revision 2.1 | Revision 2's rows | The round's rows |
|---|---:|---:|---:|---:|---:|---:|
| `dayofweek(...)` | 1 of 29 | 0 of 18 | 1 of 29 | **38 of 42** | 0 | 43 |
| the date part `dow` (`EXTRACT`, `date_part`) | 70 of 70 | 78 of 79 | 69 of 70 | 49 of 52 | 21 | 44 |
| `isodow` | 1 of 1 | 2 of 2 | 2 of 2 | 7 of 7 | 0 | 11 |
| `dayname`, `strftime` | 1 of 1 | 5 of 5 | 4 of 4 | 1 of 1 | 0 | 4 |
| another (`is_weekend`, `dow(...)`) | 0 of 1 | | | 0 of 1 | | |
| no SQL: cut at the budget | 0 of 6 | 0 of 4 | 0 of 3 | 0 of 5 | | |

- **The DuckDB gap was one function.** The date part `dow` is PostgreSQL's spelling, and every run
  numbers it as PostgreSQL does, from Sunday = 0. `dayofweek` is spelled as MySQL's function, and
  the base and Revision 2 number it as MySQL does.
  - Revision 2 learned to call `dayofweek` less (29 → 18), not to number it.
  - Revision 2.1 calls it more often than the base (42) and numbers it right. The round's rows
    name it in their convention sentence ("`dayofweek(date)` and `EXTRACT(DOW FROM date)` return 0
    for Sunday") and use it in 43 of their 102 SQL answers.
- **Revision 2.1's 6 wrong DuckDB answers:**
  - MySQL's numbers, 5 times:
    - `dayofweek(...) IN (1, 7)` for the weekend, twice;
    - Tuesday as `dayofweek(...) = 3`;
    - Friday as `EXTRACT(DOW ...) = 6`;
    - Wednesday as `EXTRACT(DOW ...) = 4`.
  - ISO's weekend in `dayofweek`, once: `IN (6, 7)`.

  The date part was wrong twice in Revision 2.1's 52 calls, and twice in the other three runs'
  219 together (Fisher p = 0.17).
- **Its 7 non-answers:**
  - **Two engine errors,** both with the right numbers: `extract('dow', ts)` (DuckDB's `EXTRACT`
    takes `FROM`) and `dow(ts)` (no such function).
  - **Five replies ran to the budget.** Three loop on a right query, writing it out again and
    again without closing their reasoning. Two go back and forth between `isodow` and `dayofweek`
    with Sunday = 1. The other three runs cut 6, 4 and 3 DuckDB replies.

### ClickHouse and the guards

- **ClickHouse's two misses are both Monday to Friday:**
  - `dayofweek(ts) BETWEEN 2 AND 6`, MySQL's working week;
  - `toDayOfWeek(ts) <= 6`, a mixed one: Saturday is 6 in ISO's numbering and Friday is 6 in
    MySQL's. It states no Sunday = 1.

  Revision 2 got all 24 workweek items in its window; 24 → 22 is p = 0.5.
- **MySQL is the regression check for DuckDB's rows,** and it holds. Its replies call
  `DAYOFWEEK` in all 24 items: `IN (1, 7)` for the weekend and `BETWEEN 2 AND 6` for the working
  week, every one right.
- **PostgreSQL:** Revision 2.1's one miss writes `EXTRACT(DOW ...) IN (1, 7)`, MySQL's weekend in
  PostgreSQL's Sunday-0 numbering.
  - Revision 2 wrote the same twice in this cell.
  - Base 1 wrote MySQL's working week once (`BETWEEN 2 AND 6`).

  Still no evidence of a regression; it stays on the list to watch.

### The old belief, in the traces

Target traces that state Sunday = 1 somewhere (`target_a_hints` doubts):

| Dialect | n | Base 1 | Revision 2 | Base 2 | Revision 2.1 |
|---|---:|---:|---:|---:|---:|
| ClickHouse | 48 | 47 | 21 | 48 | 11 |
| DuckDB | 108 | 74 | 66 | 85 | 47 |

- **Revision 2 against Revision 2.1, paired:** 51 traces state it for Revision 2 alone, and 22
  for Revision 2.1 alone (p = 0.0009).
- **The A/A:** 14 against 26 (p = 0.08).
- The training dropped every doubting row (decision 2). The round added 180 rows without a doubt,
  and the doubt went down with them.

## The Target A test

These are Revision 2's 228 items (2026-10-06): 16 cells, with the same seeds.
- **Its target** is ClickHouse's weekday and weekend, 60 items.
- **The round's DuckDB rows** show in DuckDB's weekday and weekend cells, 24 items among the other
  cells.

| Cell | n | Base 1 | Revision 2 | Base 2 | Revision 2.1 | p, base 2 vs 2.1 | p, 2 vs 2.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Target: ClickHouse weekday and weekend** | 60 | 10 | 58 | 12 | **57** | 7e-13 | 1 |
| ClickHouse weekday | 36 | 9 | 34 | 11 | 34 | 1.5e-6 | 1 |
| ClickHouse weekend | 24 | 1 | 24 | 1 | 23 | 5e-7 | 1 |
| **The 14 other cells** | 168 | 147 | 156 | 155 | 156 | 1 | 1 |
| … weekday numbering | 36 | 30 | 31 | 31 | 35 | 0.12 | 0.22 |
| … weekend flag | 36 | 28 | 30 | 31 | 29 | 0.62 | 1 |
| … timezone direction | 48 | 42 | 48 | 45 | 47 | 0.62 | 1 |
| … month bucket | 48 | 47 | 47 | 48 | 45 | 0.25 | 0.5 |
| DuckDB weekday | 12 | 8 | 8 | 7 | 11 | 0.12 | 0.38 |
| DuckDB weekend | 12 | 7 | 8 | 8 | 8 | 1 | 1 |
| DuckDB month | 12 | 11 | 11 | 12 | 9 | 0.25 | 0.5 |
| PostgreSQL weekend | 12 | 10 | 10 | 11 | 9 | 0.5 | 1 |
| PostgreSQL timezone | 12 | 9 | 12 | 11 | 11 | 1 | 1 |

The other 9 cells are 12 of 12 for Revision 2.1. Every cell is in the JSON.

- **Statuses:**
  - Revision 2.1: 213 verified (all held on the recheck), 5 wrong, 2 engine errors, 8
    unfinished.
  - Base 2: 167 verified (all held), 54 wrong, 1 engine error, 6 unfinished.
- **A/A, base 1 against base 2:**
  - target 10 → 12 (5 against 7);
  - the other cells 147 → 155 (7 against 15, p = 0.13).

  The base's other cells moved by 8 between windows, almost as far as Revision 2 moved them over
  its base (147 → 156). That gain was within the base's own spread, as its report said (p = 0.11).
- **The old belief:** Sunday = 1 is stated in 20 of Revision 2.1's 60 ClickHouse traces.
  - Revision 2: 29 (17 for Revision 2 alone, 8 for Revision 2.1 alone; p = 0.11).
  - The base: 59 and 60.

### What Revision 2.1 misses

- **ClickHouse, 57 of 60:**
  - Thursday as `toDayOfWeek(...) = 5`: MySQL's Thursday, the miss both of Revision 2's were.
  - Two replies ran to the budget:
    - a Thursday item that Revision 2 and this window's base miss too. The reply proposes `= 5`
      and answers itself "No." 379 times, never settling;
    - a weekend item: a doubt loop that states Sunday = 1 and then ISO, about 130 times each.
- **DuckDB's weekday, 11 of 12.** The one miss is `extract('dow', ts)`, which DuckDB rejects (its
  `EXTRACT` takes `FROM`). The same slip as on the round's test.
- **DuckDB's weekend, 8 of 12, as for Revision 2:**
  - `dayofweek(...) IN (6, 7)` twice, with no doubt stated. That is ISO's weekend in a function
    that numbers Sunday 0.
  - MySQL's `IN (1, 7)` once.
  - `is_weekend(...)`, a function DuckDB doesn't have, once.

  Over both tests, `dayofweek(...) IN (6, 7)` comes from Revision 2.1 3 times, from Revision 2 once,
  and never from the base. The round's rows name `isodow` (ISO's Saturday 6 and Sunday 7) in the
  same sentence as `dayofweek`, and the two numberings sometimes swap. It is a new mistake, and a
  small one.
- **`dayofweek` here: 6 of 9 right** (Revision 2: 0 of 5; base 2: 0 of 6). Over both tests it is
  44 of 51, against Revision 2's 0 of 23.
- **DuckDB's month, 9 of 12, and PostgreSQL's weekend, 9 of 12.** All six misses ran to the budget.
  - Five loop on a right query (`EXTRACT(MONTH FROM ts) = 3`, `EXTRACT(DOW FROM ts) IN (0, 6)`),
    writing it out until the budget ends.
  - One loops on MySQL's `EXTRACT(DOW ...) IN (1, 7)`.

  Neither cell's drop is significant (p = 0.25 and 0.5). These are the same loops as above, landing
  on cells the round never trained.
- **PostgreSQL's timezone, 11 of 12:** one reply subtracts the offset, the direction trap.

### Loops, over both tests

Replies cut at 12,288 tokens, of the 432 items each run answered:

| | Base 1 | Revision 2 | Base 2 | Revision 2.1 |
|---|---:|---:|---:|---:|
| Cut at the budget | 15 | 10 | 9 | 13 |
| … writing a finished query out again and again | 14 | 10 | 8 | 10 |
| … other loops (doubt, rejection) | 1 | 0 | 1 | 3 |
| Misses, all causes | 153 | 40 | 138 | 31 |

- **No run loops significantly more than another** (Revision 2.1 against base 2: 13 against 9, p
  = 0.52). The base loops like this on its own, and the mini-battery's calibration measured it.
- **As the other misses fall, the loops become a large share:** 13 of Revision 2.1's 31 misses,
  against 9 of the base's 138.
- **Revision 2.1's three other loops are all on the weekday convention:** two doubt loops (Sunday
  = 1 against ISO) and the rejection loop. Each base run had one other loop, on a timezone item.

## Reading it

- **The round did what it was for in DuckDB.** `dayofweek` now counts from Sunday 0: 44 of 51
  replies over the two tests, against Revision 2's 0 of 23.
  - Against this window's base, DuckDB's cells gain on both tests: 76 → 95 of 108 (p = 0.003) and
    15 → 19 of 24.
  - Against Revision 2, across windows, the gain is 85 → 95 and 16 → 19. Neither is significant
    on its own (p = 0.10 and 0.55).
  - The function table says why the totals move less than the function does. Most DuckDB replies
    never called `dayofweek`; they called the date part, which every run already got right.
- **Nothing the round could break broke.**
  - MySQL's `DAYOFWEEK` keeps Sunday 1 in every reply.
  - ClickHouse stays at its ceiling on both tests (46 of 48 and 57 of 60).
  - The Target A test's other cells hold (156 of 168, as for Revision 2).
  - MySQL's numbers in PostgreSQL turn up in the base's replies too: base 1 wrote `IN (1, 7)` for
    a weekend once and `BETWEEN 2 AND 6` once, across the two tests.
- **What is left is mostly loops and a few mixed numberings.**
  - 13 of the 31 misses are replies that never close their reasoning, mostly on a query they have
    already written. The base does this as often, so the training neither caused it nor cured it.
  - The rest are scattered:
    - MySQL's numbers in DuckDB, 6 times over the two tests;
    - ISO's weekend in `dayofweek`, 3 times;
    - `extract('dow', ts)`, twice.
- **These tests are the near transfer:** single SQL queries on unseen tables, in the training's
  wording. On them, the round's ClickHouse rows had no room to show, since Revision 2 was already
  at the ceiling.
- **The probe and dsbench are the far transfer,** and they run next in this window chain. dsbench's
  `da_weekend_delay`, a ClickHouse weekend in a multi-step agent run, is the problem the round's
  ClickHouse rows are for (Revision 2: 1 of 5). Whether Revision 2.1 replaces Revision 2 waits on
  them.
