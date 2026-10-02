# Target C volume run: 154 rows from the base as the agent

ADR-004 Revision 2, action item 5. Run 2026-10-01/02 on dashi, in one GPU window, with
`patches/target_c_volume_*.sh`: the generator's quota mode (`sftgen/ml_delivery_trajectories.py
--quota 22`), 22 rows a family at 8,192 tokens. Every trajectory was then measured on the box with
`patches/target-c-pilot/measure_trajectories.py`. Outputs:
- the raw outputs are in `~/benchlab/runs/2026-10-01-qwen36-target-c-volume/` and in
  `data/sft/rev2_target_c/`, which is git-ignored;
- the files' hashes are in `data/sft/rev2_target_c_manifest.json`;
- the numbers are in `20261001-target-c-volume-run.json`.

## Summary

- **154 rows, the slice the mixture asked for.**
  - Each family has its 22, except two: upsell_join has 21, because the time limit stopped new
    runs, and energy_load has 23.
  - The rows are 802,944 tokens, 378,554 of them labelled (47%). That is under the mixture
    table's 1.0M tokens for Target C.
- **The base passes 275 of 282 runs (97.5%).**
  - Every miss is energy_load's: 6 wrong, 1 out of steps.
  - There were no model or harness errors, and no dataset failed its own oracle.
  - Every turn carries reasoning, and `thinking_record` accepts all 282 trajectories.
- **The 8,192-token block cost most of the window.**
  - 121 runs passed the oracle but ran over the block.
  - 73% of the slot time went into runs that weren't kept.
  - In the three hard families, the block keeps a third of the passing loops, and the short ones.
    credit_leak's passing loops run a median of 10,532 tokens; its kept ones, 4,746.
- **At 16,384 tokens, 247 of the 275 passing loops would train** (1.87M tokens, 0.93M labelled).
  For credit_leak that is 58 of 71 instead of 22. The rows are already generated: `failed.jsonl`
  keeps them, with `oracle_passed: true`.
- **The agent's own login held** (ADR-003 §7).
  - No run's tool calls named an answer key or the admin's login, and none saw a key in a
    listing.
  - No call was refused for want of a grant. Every ClickHouse error is the model's own SQL.
- **A fifth of the kept rows' assistant text sits in turns whose call failed** (20.5%). That is
  141 of 1,158 turns, in 88 of the 154 rows. Decision 7 is whether those turns train.
- **Found: two kinds of run the selection judged by the wrong measure.** None of them was kept
  here. The selection now drops the first kind and counts the second.
  - Runs that used up their steps after writing a passing table: 2.
  - Rows ending on tool output that no request carried: 3.

## What was run

- **Tasks:** the 7 families, from run index 200 (Gate 2 used 0-62, the pilot 101-103). Every
  dataset passed its own oracle before the agent saw it. Every task allows 24 steps.
- **Serving:** the base (APEX I-Mini, no LoRA), thinking on, with Qwen's sampling (temperature
  0.6, top-p 0.95, top-k 20). 8 slots of 196,608 tokens, at most 8,192 tokens a turn.
- **The agent loop** ran on the Mac, beside its sandbox, through an SSH tunnel that restarts
  itself. Each run had its own ClickHouse database, working directory and login. The login had no
  `aviation` grant.
- **Quota mode:** a free slot goes to the family furthest from its 22 rows, counting each of its
  running loops at its keep rate so far.
- **The window:**
  - production was stopped at about 21:37 CEST. OCR had started by then: the window stopped it
    once it had loaded, and restored it at the end;
  - generation ran 21:38-02:19, 4 hours 40 minutes;
  - no run started after 02:08, the hold's deadline less 30 minutes;
  - the Mac released the hold at 02:19, 19 minutes before its 5-hour limit.
- **The box's clock runs 7,131 seconds behind the Mac's** (NTP off). The Mac converted the hold's
  deadline across the difference.

## Results

| Family | Runs | Passed | Kept | Passing loop, median tokens | Kept, median | Passing within 16,384 |
|---|---:|---:|---:|---:|---:|---:|
| widget_defect | 22 | 22 | 22 | 3,834 | 3,834 | 22 |
| churn_rare | 22 | 22 | 22 | 4,143 | 4,143 | 22 |
| ticket_route | 23 | 23 | 22 | 3,774 | 3,718 | 23 |
| delivery_time | 23 | 23 | 22 | 5,245 | 5,190 | 23 |
| upsell_join | 53 | 53 | 21 | 9,400 | 7,320 | 49 |
| energy_load | 68 | 61 | 23 | 9,560 | 6,598 | 50 |
| credit_leak | 71 | 71 | 22 | 10,532 | 4,746 | 58 |
| **All** | **282** | **275** | **154** | **7,603** | **4,856** | **247** |

- **The pilot's keep rates held.** The four easy families keep nearly every run. The three hard
  ones keep about a third: upsell_join 40%, energy_load 34%, credit_leak 31%.
- **The misses are energy_load's trap**, the temporal train/test boundary. Against a bar of 25%
  lift over the naive forecast, the 7 misses score from -164% to +17%. One of them ran out of
  steps.
- **A kept row has a median of 7 steps (at most 17).** A loop over the block has a median of 13.

### Time

- **282 runs in 4 hours 40 minutes:** 60 an hour, with 7.8 of the 8 slots busy on average. The
  loops generated 1.40M tokens, 83 a second.
- **A run took a median of 6.1 minutes:**
  - a kept run 3.3 minutes, and 9.2 at most;
  - a run over the block 10.2 minutes;
  - the longest 49 minutes.
- **The simulation, on the pilot's run times, expected 286-336 runs in 3.4-4.1 hours.**
  - The hard families drew 192 of the 282 runs, and their runs over the block are the long ones.
  - The last runs finished 10 minutes after the last start.
  - Every run longer than 20 minutes was over the block or missed, so the 30-minute drain could
    not cut a row that would have been kept.

### The block

| Block (tokens) | Passing loops that fit | Tokens | Labelled |
|---:|---:|---:|---:|
| 8,192 | 154 | 0.80M | 0.38M |
| 12,288 | 214 | 1.41M | 0.69M |
| 16,384 | 247 | 1.87M | 0.93M |
| 24,576 | 266 | 2.23M | 1.13M |

- The 121 passing loops over 8,192 run a median of 12,388 tokens (p90 20,494, at most 47,222).
- **The block selects within a family.** In credit_leak, upsell_join and energy_load, it keeps
  the short loops, at half or two thirds of their family's median length. The pilot showed what
  that drops: both of credit_leak's loops that found the leak were over the block.

### Failed calls in the kept rows (decision 7)

- 141 of the kept rows' 1,158 assistant turns are answered by an error. They hold 20.5% of the
  assistant text (reasoning, reply and call arguments). In the pilot, about 28% of the labelled
  text sat in such turns.
- **Errors from ClickHouse, across all runs:**
  - functions under other dialects' names, 65: the standard deviation as `stddev` or `stdDev`
    (ClickHouse has `stddevPop` and `stddevSamp`), 34 times; `typeof()` (ClickHouse's is
    `toTypeName()`), 29 times;
  - syntax errors, 51: 46 are a `FORMAT` clause of the model's own, which the client's
    `FORMAT Native` then breaks;
  - a column neither aggregated nor grouped, 28;
  - a column the table doesn't have, 8;
  - refused for want of a grant, 0.
- **Errors in Python, in the kept rows:** 33 AttributeErrors, mostly the ClickHouse client's
  result API (21 calls reach for `QueryResult` or `result_df`), 14 KeyErrors and 10 ValueErrors.
- Labelled, these turns would train the base's own wrong guesses, as the pilot found.

### Found: what the selection measured wrong

1. **Runs that never stopped.**
   - upsell_join #217 and energy_load #223 used up their 24 steps after writing a passing table.
     The oracle grades the table whatever ended the run, as the measurement loop does, and the
     status was `ok`.
   - Such a row teaches tool calls that never stop, which Gate 2's battery found in its data.
   - `run_teacher_agent` now records whether the agent ended the loop itself
     (`finished`: `finish` or a final reply). `selection` drops the others as `out_of_steps`.
2. **Tool output after the last request.**
   - A row's length is the server's count for the last request, plus one token. On 239 of the 282
     trajectories it was exactly that.
   - Three end on tool output that no request carried: the two above, and energy_load #222, which
     ran out of steps and missed. Those rows are longer than the count plus one, by 56, 291 and 504
     tokens. The 504 were 521 characters of numbers, which tokenize digit by digit.
   - `selection` now adds at most a token a character, plus 16 a message, for output after the
     last request. A turn that writes the table and calls `finish` at once ends that way too.
   - Re-selected under both rules, no kept row changes. The two that ran out of steps move from
     `over_block` to `out_of_steps`.
3. **The rest of the count is as the pilot found.** On the other 40 rows, the row is the count
   or up to 27 tokens shorter, because the template rebuilds `finish`'s arguments from the parsed
   value. The generator's selection agrees with the box's measurement on all 282.

## What it means for the retrain

1. **Target C's slice is done at 8,192 tokens.**
   - It has 154 rows, 0.80M tokens and 0.38M labelled.
   - That is 8% of a 10M-token mixture, against the table's 10%. The other 0.2M go to the other
     buckets, or come from a longer block.
   - upsell_join's missing row is not worth a window.
2. **The block decision now has its numbers.**
   - At 16,384 tokens, Target C would have 247 rows and 1.87M tokens, and 150 drawn by family
     would make about 1.1M. Its rows would include the loops that reason longest in the hard
     families.
   - 16,384 needs new kernel keys and a memory check, as 4,096 did. The data needs no new window.
   - If the block stays at 8,192, the next run could stop a loop once its context passes the block.
     73% of this window's slot time went into runs that weren't kept.
3. **Decision 7 (failed turns):** a fifth of the kept text is in them. The proposal, to mask
   them and keep them as context, stands. Unmasked, the commonest lessons would be other
   dialects' function names, a `FORMAT` clause the client can't take, and the wrong client API.

**Limits:**
- One window and one sample per dataset.
- The keep rates come from 22 to 71 runs a family.
- The failed-turn share counts characters of assistant text, with errors matched by pattern in
  the tool output.
- The 16,384 numbers count rows that pass the oracle and fit; a retrain at that length would also
  need its packing and memory checked.
