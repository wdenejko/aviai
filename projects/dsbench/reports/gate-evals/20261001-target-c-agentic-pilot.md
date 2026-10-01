# Target C agentic pilot: the base as the agent, thinking on

ADR-004 Revision 2, action item 5. Run 2026-10-01 on dashi, in one GPU window. The code is
`sftgen/ml_delivery_trajectories.py --thinking`, measured with
`patches/target-c-pilot/measure_trajectories.py`. The raw outputs are in
`~/benchlab/runs/2026-10-01-qwen36-target-c-pilot/`, and the numbers are in
`20261001-target-c-agentic-pilot.json`.

## Summary

- **The base needs no teacher.** 20 of 21 runs pass the oracle, and every family passes at least
  2 of 3. All 215 assistant turns carry reasoning, and `thinking_record` rejects none of the 21
  trajectories. Principle 2's exception, a teacher for Target C if the base's yield is too low, is
  not needed.
- **The 8,192-token block keeps 14 of the 20.** A passing loop runs a median of 5,715 tokens, and
  6 run over the block, up to 17,399. The 6 come from the hard families: credit_leak and
  upsell_join keep 1 of 3 each, and energy_load 1 of its 2 passes.
- **Most of the length is code, not reasoning.** Of a passing loop's characters:
  - reasoning is 18%, a median of 291 characters a turn;
  - tool-call arguments are 56%: `run_python` starts a fresh process on every call, so each fix
    resends the whole script;
  - tool output is 25%.
- **The block is a filter, and it selects.**
  - The loops it drops failed more calls: 2.67 erroring turns each, against 1.29 in the kept ones.
  - It drops both credit_leak loops that found the leak. Each trained on `collections_flag`,
    checked it, and reasoned its way to dropping it. The kept loop left the column out from its
    first model and never said why.
- **About 28% of what a passing loop would train sits in turns whose call failed.** 16 of the 20
  have at least one such turn.
  - The commonest error is reading a ClickHouse result into pandas: 16 of 41 errors, in 9 runs
    (`QueryResult.result_df`, `pd.read_sql` on the client).
  - Labelled, those turns train the base's own wrong guesses about the API.
- **The one miss is the trap its family sets.**
  - energy_load #103 forward-filled lag features across the 1,000-hour test horizon and validated
    one step ahead: MAE 1.77 there, against 13.79 on the test, where the naive forecast scores
    14.16.
  - Its reasoning named the problem twice. Both times it kept the model because the validation
    number looked good.
- **Found on the way: boolean arguments trained as Python's `True`.**
  - The template prints a top-level argument with Python's `str`. Served, llama-server's grammar
    held the base to JSON's `true`.
  - Fixed in `tokenize_masked._as_written`. 30 of the tool rows' 295 fit prompts call a tool with
    a boolean parameter.
- **For the volume run:**
  - keep the 8,192 block and fill a quota per family, about 300 runs and 3.5 hours of box time
    for 150 rows;
  - decide whether turns whose call failed train (proposed: mask them; they stay as context).

## What was run

- **Tasks:** the 7 families x 3 datasets, run indices 101-103 (Gate 2 used 0-62). The oracle
  passed all 21 datasets before the window, with no model (`--oracle-only`). Every task allows 24
  steps.
- **Serving:** the base (APEX I-Mini, no LoRA), thinking on, with Qwen's sampling (temperature 0.6,
  top-p 0.95, top-k 20). 8 slots of 196,608 tokens, at most 8,192 tokens a turn. The smoke test
  confirmed the thinking engaged.
- **The loop** ran on the Mac, through an SSH tunnel, 8 runs at once. Each run had its own
  ClickHouse database and working directory. Each request resent the earlier turns' reasoning, as
  the training row will show it.
- **The window:** 18.8 minutes from production down to OCR restored, 17.3 of them generating. The
  thermal governor never stepped in. OCR was stopped once loaded and restored at the end.
- **Measurement:** every trajectory went through `thinking_record` on the box, with the tokenizer
  matched to llama.cpp's. A row is kept when the oracle passes, `thinking_record` accepts it, and
  it fits 8,192 tokens.

## Results

Runs 101, 102 and 103 of each family:

| Family | Passed | Kept | Tokens | Steps |
|---|---:|---:|---|---|
| widget_defect | 3 | 3 | 4,143 · 4,048 · 6,322 | 6 · 7 · 6 |
| delivery_time | 3 | 2 | 4,192 · 7,636 · **8,964** | 6 · 11 · 12 |
| churn_rare | 3 | 3 | 5,749 · 4,443 · 4,110 | 12 · 7 · 6 |
| ticket_route | 3 | 3 | 4,335 · 4,963 · 4,305 | 8 · 9 · 10 |
| energy_load | 2 | 1 | 5,680 · **9,540** · *24,336* | 6 · 7 · 15 |
| credit_leak | 3 | 1 | **12,650** · 4,704 · **13,264** | 18 · 7 · 17 |
| upsell_join | 3 | 1 | 7,875 · **17,399** · **9,561** | 13 · 17 · 15 |

Bold: passed but over 8,192 tokens. Italic: the miss.

- The 14 kept rows average 5,179 tokens, of which 2,547 are labelled.
- A passing loop's turns are 51% labelled.
- A passing run took a median of 227 seconds. The miss took 655.

### Turns and reasoning

- **215 assistant turns.** Every one ended in a tool call, and every run ended by calling `finish`.
  No turn reached the 8,192-token limit; the longest was 2,673 tokens.
- **The reasoning is short.** It runs a median of 291 characters a turn (p90 609): a plan, or a
  reading of the last output. For example, after a traceback: "Let me fix the attribute - it's
  `result_set` not `result_df`."
- **Gate 2's loops ran a median of 2,820 tokens** (Ling, with no reasoning). The base's run twice
  that, and reasoning is under a fifth of it.

### Where the length comes from

| Part of a passing loop | Characters |
|---|---:|
| tool-call arguments (code and SQL) | 56% |
| tool output | 25% |
| reasoning | 18% |
| reply text | 1% |

- **The code is resent whole.** `run_python` runs each call in a new process, so a script holds
  its imports, the client, the loading, the features and the model. A fix resends all of it. In
  credit_leak #103, 7 of 12 `run_python` calls open the same way as an earlier call.
- **Tool output** is capped at 6,000 characters by the harness. The largest reply hit the cap: a
  `GROUP BY` with no `LIMIT`, in upsell_join #102.
- **Loops check their work.** A passing loop first writes its table at a median of step 5 (3 to
  9), then spends a median of 3 more steps checking it, up to 13. That tail is not waste to cut:
  - credit_leak #101 wrote its first table at step 5, from a model that used the flag;
  - a check two steps later found the leak;
  - step 9 rewrote the table without it.

### The block selects

- **Clean loops fit; loops with errors don't.** The kept loops average 1.29 erroring turns, and
  the 6 dropped passes 2.67.
- **credit_leak shows what that costs:**
  - **#102 is kept** (4,704 tokens). It left `collections_flag` out of its first model, and its
    reasoning never mentions the column.
  - **#101 is dropped** (12,650 tokens). Its first models used the flag. A crosstab then showed
    the flag equal to the label, and the reasoning followed: "This is data leakage!", then "I
    should NOT use collections_flag".
  - **#103 is dropped** (13,264 tokens). The model put an importance of 1.0 on the flag. The
    reasoning wavered ("But maybe that's the right answer") before dropping it.
- **So at 8,192, credit_leak's rows show the right action without the reasoning for it.** The two
  loops that reason through the leak, the family's lesson, are the ones the block drops.

### Errors

41 errors in the 21 runs, counted from the last line of each traceback or SQL error:

| Kind | Errors | Runs |
|---|---:|---:|
| reading a query result into pandas | 16 | 9 |
| ClickHouse function names | 6 | 5 |
| ordinary pandas and scikit-learn mistakes | 19 | 10 |

- **The kinds.**
  - Reading a result: `QueryResult` attributes that don't exist (`result_df`, `data`, `names`),
    and `pd.read_sql` or `pd.DataFrame` on the client's objects.
  - Function names: `typeof`, `toType` and `stddev`, which ClickHouse spells `toTypeName` and
    `stddevPop`.
  - The rest: shapes, missing imports, wrong arguments.
- **Recovery works.** 16 of the 20 passing loops have at least one erroring turn and recover from
  it.
- **The errors are a large share of what would train.** 28% of the passing loops' reasoning and
  arguments sits in those turns.
- **The system prompt shows how to build the client, not how to read a result.** 11 of the 21
  runs read with `client.query_df`; the errors are the other guesses.

### The miss: energy_load #103

- **What it built.** Lags of 1 to 24 hours, filled forward across the test period, which starts
  where the training data ends and runs 1,000 hours.
- **How it validated.** On the last 1,000 training hours, with lags built from true values. That
  is one step ahead, with no forecast horizon. Validation MAE was 1.77; on the test it was 13.79,
  against the naive forecast's 14.16, a 2.7% lift where the bar is 25%.
- **What its reasoning said.**
  - At step 8: "ffill() would carry forward the last training value, which is incorrect". It
    tried other features, then at step 11 chose the lag model by its validation MAE.
  - At step 12 it worked out that lag 1 for hour 4001 needs a prediction ("a recursive
    forecasting problem"), and kept the model: "looking at the validation MAE of 1.77, this
    seems very good".
- **What it means.** A validation number that cannot fail overruled the reasoning. Ling's Gate 2
  misses on this family had the same cause: lags with no values across the horizon, and a
  validation that couldn't see it (energy_load at 66% yield). The oracle catches it, and that is
  what the family is for.
- **The two passes** used hour, weekday and temperature features with no lags (lifts of 64% and
  77%).

### The training row is the generated loop

Compared with the server's own counts:

- **The row's length.** On 16 of 21 runs, the row is one token longer than the loop's last
  request (prompt plus completion): the newline the template writes after the final
  `<|im_end|>`.
- **The labelled tokens.** On the same 16, they equal the completion tokens the server counted.
- **The other 5.** Each differs by 1 to 12 tokens. Each passed `finish` an `answer` that is a
  bool (3), a dict or a list of floats; every run whose `answer` was a number matches. The server
  returns that value parsed, not the tokens the base generated, so the row rebuilds the call from
  the value.

So a volume run can apply the 8,192 cap from the server's counts, without measuring on the box: no
row is more than one token longer than its loop's last request.

### Found: booleans train as Python's `True`

- **The cause.** The template prints a top-level argument with `tojson` if it is a mapping or a
  list, and with Jinja's `string` filter otherwise. That filter is Python's `str`, so `true`
  renders `True` and `null` renders `None`.
- **What the base wrote.** Served, llama-server's grammar holds a non-string parameter to its JSON
  schema, so the base wrote `true`. The label would teach it `True`.
- **The fix.** `tool_arguments_as_objects` now passes a top-level boolean or null as its JSON text
  (`_as_written`).
  - Tested: `test_a_boolean_or_null_argument_trains_as_the_json_the_model_wrote`.
  - Re-measured on the box: `finish(answer=true)` now renders `true`, and no row's tokens,
    labels or kept status changed.
- **Where it matters.** Here, only the `finish` call's free-form `answer`. In the tool rows, 30
  of the 295 fit prompts call a tool with a boolean parameter:
  - `set_lights.on` (10);
  - `find_restaurants.open_now` (10);
  - `list_directory.include_hidden` (10).

## What it means for the retrain

1. **No teacher for Target C.** The base passes 20 of 21, with reasoning in every turn.
2. **Volume at 8,192 tokens, with a quota per family (proposed).**
   - At the pilot's keep rates, 150 rows take about 300 runs. That assumes 22 rows a family,
     with credit_leak, energy_load and upsell_join keeping about 1 run in 3.
   - That is about 3.5 hours of box time, at about 100 runs an hour with 8 slots busy (5.7 were
     busy on average here, because of the tail).
   - Without quotas, about 225 runs and 2.2 hours would fill 150 rows, but nearly two thirds of
     them would come from the three easy families.
   - 150 rows are about 0.8M tokens at the kept rows' lengths, under the mixture table's 1.0M.
   - **The alternative is a 16,384-token step.** It would keep 19 of the 20 passing loops,
     credit_leak's reasoning included. But it needs new kernel keys, as 4,096 did, and a memory
     check.
3. **A new decision: whether turns whose call failed train.**
   - Proposed: mask their loss and keep them as context. The fix that follows then trains with
     its cause in view.
   - It costs the loss on about 28% of the passing loops' labelled text.
   - The oracle checks the delivered table, not each step. An erroring call is, by the execution
     record, a wrong step.
   - Not proposed: stating `client.query_df` in the system prompt. It would cut the commonest
     error, but the rows would then use the API because the prompt gave it, and the eval prompts
     don't.
4. **Generator changes before volume.**
   - A quota mode: run each family until its rows fill, kept meaning the oracle passed and the
     server's count fits 8,192.
   - If decision 3 goes that way, a per-turn mark that `thinking_record` reads to leave a turn
     unlabelled.

**Limits:** 3 runs per family, so the per-family keep rates are rough: 1 of 3 is consistent with
anything from rare to most. Each run had one sampling seed. The error kinds come from a pattern
on the last line of each error. The selection effect rests on credit_leak's three loops.
