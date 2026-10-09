# Gate-2 acceptance battery: the Gate-2 adapter vs its base, paired per item

**Date:** 2026-09-24/25, GPQA 2026-09-28 · raw numbers: `20260924-gate2-battery.json` · harness:
`src/dsbench/battery/` · box launchers: `patches/battery_*.sh` · run directory, logs, the raw
pi-run JSON of the owner's suite (`pi-runs/`; summaries in `reports/agentic-runs/`) and the
one-off analysis scripts behind the numbers below (`analysis/adhoc_numbers.py` and friends):
`~/benchlab/runs/2026-09-24-gate2-battery/` on dashi

**Status: final (2026-09-25; GPQA, gated until then, added 2026-09-28). Gate 2 is not passed, at
full scale or at half scale; see "Gate-2 decision" and "What the retrain has to change".**

## What ADR-001 asks, and how it was measured

Gate 2 (item 3) accepts the adapter if it **gains on at least two target benchmarks** (DS-1000,
BIRD-dev, the owner's dsbench suite, tool-call format tests) while **regressing no more than 1 point**
on IFEval, MMLU-Pro and GPQA, **no more than 2 points** on LiveCodeBench and HumanEval+, and losing
**no more than 5 points of MTP draft acceptance**.

One `llama-server` (production `build-v2`, Vulkan) loads the I-Mini base GGUF and the Gate-2 LoRA
once. Every request names its state: LoRA scale 0 (base) or 1 (adapter). The binary, quantization,
chat template and sampler are therefore shared, and the only difference between the two columns is
the adapter. Decoding is greedy with thinking off, the regime the adapter was trained for. Absolute
scores therefore sit below Qwen's published thinking-mode numbers; the question here is the
difference, not the level.

Each item is scored in both states. The test is an exact McNemar test on the discordant pairs, and
the 95% interval is the matching exact interval (Clopper-Pearson on the gain share, mapped to
percentage points), so the interval excludes zero exactly when p < 0.05. A third pass, `base_rep`,
runs the base again: its flip count is the harness's own noise on that benchmark.

| Benchmark | Items | Subset | Prompt protocol | Scorer (pinned) |
|---|---:|---|---|---|
| HumanEval+ | 164 | all | EvalPlus chat instruction | the HF `evalplus/humanevalplus` test program |
| DS-1000 | 1000 | all | DS-1000 `run_openai.py` (system prompt, stops) | DS-1000's `code_context` test program |
| IFEval | 541 | all | the prompt as given | Google `instruction_following_eval` |
| MMLU-Pro | 1400 | 100 per category, seeded | official CoT instruction, zero-shot | official regex chain |
| GPQA | 198 | diamond | simple-evals multiple-choice template, options shuffled per question (seeded) | simple-evals' `Answer: X` line, the last one |
| LiveCodeBench | 342 | v5+v6 additions (2024-09-22 to 2025-04-06) | LCB generic chat prompt | LCB `testing_util.run_test` |
| BIRD-dev | 1534 | all | DDL + 3 example rows per table + evidence | set-equality execution accuracy (30 s) |
| BFCL | 1240 | simple, multiple, parallel, parallel-multiple, irrelevance | BFCL function-calling conversion | BFCL `ast_checker` |
| dsbench (owner) | 23 | all, k = 5 via the pi harness | the Gate-0 protocol | the suite's own ClickHouse end-state oracle |

## Harness validation (ADR-001 Gate 1, item 5)

Before any adapter number counts, the harness has to be shown to measure the model and nothing else.

- **"Scale 0" is the base model.** Eight fixed prompts, greedy: a server started *without* the
  adapter, the battery server at scale 0, then scale 1, then scale 0 again on the same slots.
  Scale 0 matched the bare base 8/8 and itself after a scale switch 8/8 (so a slot's prompt cache
  never carries KV across states), and scale 1 differed 8/8.
- **Reference solutions pass.** DS-1000 1000/1000 (after baking into the sandbox image the five
  public datasets 15 problems download at test time); BIRD 1532/1534 (two gold queries exceed
  BIRD's own 30-second limit); HumanEval+ 163/164 (the Hugging Face copy of HumanEval/32's test
  calls `_poly(*candidate(*inp), inp)` with its arguments swapped, which fails any solution). The
  three are excluded as unmeasurable.
- **Scorers on synthetic answers.** IFEval passes an obedient reply on all three instructions of
  prompt 1000 and fails a disobedient one on all three; BFCL accepts the correct call, rejects a
  wrong argument, accepts parallel calls in either order and a call-free irrelevance reply;
  LiveCodeBench returns `wrong` (0/42 tests), not `error`, for a do-nothing solution.
- **The A/A noise floor is not small.** Greedy decoding on the batched server is not
  deterministic: which requests share a batch changes the Vulkan kernel path (mat-vec kernels are
  instantiated per column count) and hence float summation order, and an answer can diverge at one
  token. Between two identical base passes, HumanEval+ flipped 5 of 163 items, DS-1000 74 of 1000
  (47.1% vs 47.9%) and IFEval 45 of 541 (81.7% vs 81.1%): the longer the replies, the more they
  diverge. McNemar handles symmetric noise correctly, but it costs power, and any adapter effect has
  to be read against it.
- **Contamination.** ADR-001 asked for the mixture to be decontaminated against every eval set; it
  was only checked against dsbench's own prompts. A 13-gram check of every battery item against
  the pilot and Gate-2 mixtures (`contamination.py`) found one item over the 20%-of-grams "strong"
  line (BFCL `multiple_1`, whose short question contains a generic sentence) and 44 with any
  shared 13-gram, all generic: counting sequences, one-hot rows, textbook Fibonacci and digit-name
  code, the definition of a subsequence. GPQA's 198 questions, checked when access came, share
  none. The battery does not measure memorization.

## Results

Full scale (LoRA 1.0) against the base, paired per item (`report.py`; half scale is in the step-4
section):

| Benchmark | ADR role | n | Base | Adapter | Δ pp (95% CI) | +/− | p | A/A flips | Verdict |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| DS-1000 | target | 1000 | 47.1 | 47.8 | +0.7 (−2.3, +3.6) | +110/−103 | 0.68 | 74/1000 | no gain |
| BIRD-dev | target | 1532 | 60.1 | 51.0 | −9.1 (−11.0, −6.9) | +84/−223 | 1e-15 | – | regression |
| dsbench (owner, majority of 5 runs) | target | 23 | 78.3 | 82.6 | +4.3 (−10.6, +12.8) | +2/−1 | 1.0 | – | no gain |
| BFCL AST | target | 1000 | 88.5 | 91.5 | +3.0 (+0.9, +4.9) | +66/−36 | 0.004 | – | gain |
| IFEval (prompt, strict) | ≤ 1 pt | 541 | 81.7 | 77.1 | −4.6 (−8.1, −0.9) | +36/−61 | 0.014 | 45/541 | fail |
| MMLU-Pro | ≤ 1 pt | 1400 | 80.8 | 77.7 | −3.1 (−4.8, −1.3) | +58/−101 | 0.0008 | – | fail |
| GPQA (diamond) | ≤ 1 pt | 198 | 59.1 | 67.7 | +8.6 (+1.8, +14.0) | +29/−12 | 0.012 | – | pass (confirmed) |
| LiveCodeBench | ≤ 2 pt | 342 | 37.4 | 30.1 | −7.3 (−10.6, −3.1) | +13/−38 | 0.0006 | – | fail |
| HumanEval+ | ≤ 2 pt | 163 | 90.8 | 85.9 | −4.9 (−8.4, +0.5) | +4/−12 | 0.077 | 5/163 | fail (n.s.) |
| BFCL irrelevance | none | 240 | 89.2 | 67.1 | −22.1 (−24.5, −17.3) | +4/−57 | 5e-13 | – | regression |
| MTP acceptance | drop ≤ 5 pt | 200 prompts | 66.2 | 69.4 | +3.2 | | | | pass |

Accuracies are percentages; "+/−" counts the items the adapter gained and lost. A/A flips are the
items a second, identical base pass changed (only the three first-phase benchmarks got one). GPQA's
pass is the token budget, not better reasoning: 28 of its 29 gains are questions the base ran out
of tokens on (see its section).

### DS-1000: much churn, no net gain

47.1% → 47.8% (+0.7 pp, 110 gains / 103 losses, p = 0.68). The adapter changes about a fifth of
all answers, but the A/A pass moved the base by +0.8 pp with 74 flips, so the net change sits inside
the harness noise. By library the churn is not neutral for a data-science tune: Pandas 14 gains / 23
losses and SciPy 7 / 18 go backwards, while NumPy (31 / 18) and PyTorch (18 / 9) go forwards. The
adapter writes much terser answers (46 vs 103 tokens on average), which removes a class of base
failures (13 of its gains were base `IndentationError`s) and adds wrong ones.

### IFEval: a real instruction-following regression

81.7% → 77.1% prompt-level strict (−4.6 pp, 36 gains / 61 losses, p = 0.014); all four published
IFEval numbers drop by 3.5 to 5.2 points. The think-leak is not the cause here (15 closed blocks, no
open ones; on the 526 leak-free items the delta is still −3.8 pp, p = 0.04). The losses are diffuse:
length floors ("at least N words" broken 9 times, kept 3), two-response formats (6 / 1),
placeholders (5 / 0), highlighted sections (6 / 2). The adapter writes shorter replies (median 694
vs 850 characters), consistent with the training text, whose assistant turns are short (median 94
tokens, 90th percentile 255). Under greedy decoding it also falls into repetition loops more often
(24 replies hit the 4096-token limit vs 9). That is not a masking bug that left stopping untrained:
every complete assistant turn in the training blocks ends on a trained `<|im_end|>` (30,321 of
them; 2,158 more are cut by the 2048-token block edge and end in the next block). The whole
pattern is general-skill forgetting of the kind ADR-001's replay fraction was meant to prevent.

### MMLU-Pro: the adapter reasons half as long, and loses 3 points

80.8% → 77.7% (−3.1 pp, 95% CI −4.8 to −1.3, 58 gains / 101 losses, p = 0.0008), against a 1-point
limit. Twelve of the fourteen categories drop; the exceptions are computer science (80 → 88) and
physics (82 → 83), and the worst are engineering (65 → 54) and history (75 → 67). The think-leak plays
no part (12 closed blocks; −3.0 pp without them), and neither does the answer format: the adapter
ends with "the answer is (X)" more often than the base does (1,352 vs 1,317 of 1,400).

What changed is the length of the reasoning. The prompt asks for step-by-step thinking, and the
adapter writes about half as much of it: median 386 tokens against 728, and per item 0.46 times
the base's length on geometric average. Where it loses an item, the base had reasoned for a median
of 1,495 tokens and the adapter for 504. The same brevity explains most of its gains: in 20 of its
58 gains the base ran into the 4096-token limit. This is the IFEval finding again (shorter
replies), and DS-1000's (46 vs 103 tokens), on a benchmark where the length of the reasoning is
the method.

### GPQA: a gain made of the token budget

59.1% → 67.7% (+8.6 pp, 95% CI +1.8 to +14.0, 29 gains / 12 losses, p = 0.012). By the ADR's rule
that is a pass, and GPQA is the only general benchmark the adapter moves up. The cause is the same
brevity as MMLU-Pro's; it nets out the other way because here the budget binds far more often. With
thinking off, both states reason in the reply, and at this battery's 4096-token budget the base
never reaches its answer line on 68 of the 198 questions (median reply 2,118 tokens). The adapter
reasons about a quarter less (median 1,469 tokens; 0.77 of the base's length per item, geometric
mean) and runs out on 38. Every unanswered reply in either state is one that hit the limit; none is
an answer line the scorer missed.

28 of the adapter's 29 gains are questions where the base ran out of tokens. On the 126 questions
both states answered, the base is right 90.5% of the time and the adapter 84.1% (1 gain, 9
losses, p = 0.02). That comparison selects on the states' own replies, so it is an indication,
not a clean test, but it points where MMLU-Pro does: shorter reasoning fits the budget more often
and is wrong more often. Chemistry, whose questions carry the longest calculations, holds most of
the gain (34.4 → 49.5; physics 86.0 → 88.4, biology 57.9 → 63.2 on 19 questions). No reply opens
a `<think>` block in either state. At a budget where the base finished its reasoning, the
result could flip sign: this pass says the adapter is more economical, not that it reasons better.

### LiveCodeBench: the think-leak at scale

37.4% → 30.1% (−7.3 pp, 95% CI −10.6 to −3.1, 13 gains / 38 losses, p = 0.0006), against a 2-point
limit. Medium problems carry most of it (42.3% → 26.9%; easy 85.7% → 82.1%, hard 7.8% → 3.9%).

Under this benchmark's budget (4096 tokens, thinking off) LiveCodeBench is partly a test of fitting
a solution into the budget. Even the base reasons in the visible reply on hard problems, and 176 of
its 342 replies hit the limit, 175 of them before writing any code. The adapter adds the think-leak
at a scale no other benchmark shows: 246 of its 342 replies open a `<think>` block (base: 33), and
169 never close it, so they end with no code (base: 3). Where the adapter does close its block, it
solves 64 of 77. So its reasoning isn't useless; there just isn't room for it. It isn't only the
budget, though: on the 96 items where neither state opens a block, the adapter still trails (51.0%
→ 40.6%, p = 0.03), and where both states wrote code it passes 93 of 121 against the base's 101.
A larger budget would change the absolute numbers and could narrow the gap. It was not run: at
16k tokens a single state would take roughly five hours or more on this box.

### BIRD-dev: a target benchmark goes backwards

60.1% → 51.0% execution accuracy (−9.1 pp, 84 gains / 223 losses, p = 1e-15), worst on the harder
questions (challenging 50.7% → 32.6%, moderate 50.5% → 37.6%, simple 66.3% → 60.5%). The adapter's
SQL fails to run almost four times as often (176 errors vs 47), and the errors are specific:

| SQLite error | Base | Adapter | Adapter at scale 0.5 |
|---|---:|---:|---:|
| `no such column` | 18 | 101 | 30 |
| `no such function` (adapter: `YEAR` 57, `MONTH` 2, and BIRD's evidence notation `DIVIDE`/`SUBTRACT` 4) | 10 | 63 | 37 |
| all execution errors | 47 | 176 | 77 |

Two habits, neither of them ClickHouse (0 of 1534 replies use a ClickHouse-only function, so Target
A's dialect did not bleed across). The adapter reaches for MySQL-style `YEAR(date)`, which SQLite
lacks, and it invents plausible column names instead of reading the schema it was given (selecting
`AdmsFName1` from the table that doesn't have it, filtering on a `DistrictCode` that doesn't exist).
Both habits are in the training data. The mixture's SQL pool is Gretel's synthetic text-to-SQL
(6,345 rows, a fifth of all assistant turns), and each row carries its own CREATE/INSERT context.
Run in SQLite against that context (in the sandbox), one training answer in five fails: 1,270 of
the 6,149 whose schema loads. 266 call a function SQLite lacks (`YEAR` 120, `MONTH` 24, `DATEDIFF`
22, `DATE_FORMAT` 21, ...), 404 use other MySQL or T-SQL syntax, and about 350 query a column
(~150) or a table (200) that their own schema doesn't define. SQLite's date function, `strftime`,
appears in one of Gretel's 6,345 rows; `YEAR(` appears in 202. The adapter learned what it was
shown. No
BIRD-like data (real schemas, SQLite, external evidence) was in the mixture: Gretel stood in for
ADR-001 B.4's text-to-SQL bucket, which had planned OmniSQL/SynSQL and BIRD train rows.

### BFCL: better tool calls, and tool calls where none belong

Averaged over all 1240 items the adapter looks flat (88.6% → 86.8%, p = 0.085). BFCL's own
leaderboard never averages these, and neither should this report:

| BFCL column | Items | Base | Adapter | Δ pp (95% CI) | +/− | p |
|---|---:|---:|---:|---|---:|---:|
| AST: simple, multiple, parallel, parallel-multiple | 1000 | 88.5 | 91.5 | +3.0 (+0.9, +4.9) | +66/−36 | 0.004 |
| Irrelevance: no offered tool fits the question | 240 | 89.2 | 67.1 | −22.1 (−24.5, −17.3) | +4/−57 | 5e-13 |

Every call category improves (simple +2.3, multiple +3.0, parallel +5.5, parallel-multiple +2.0),
and the output format is essentially perfect in both states (1239/1240 adapter calls parse). That
is the ADR's tool-call target, met. But when the offered functions don't fit, the adapter now
invents a call instead of answering: triangle area via `determine_body_mass_index(weight=10,
height=5)`, a quadratic "solved" with `math_sum([1, 2, 3])`. The training data explains it. Tools
appear in 896 of the mixture's 23,640 rows (Jupyter Agent 559, Target C 337), and in all 896 the
first reply is a tool call. No row offers a tool that doesn't fit, so nothing taught the adapter to
decline. The ADR sets no threshold for this, but in an agent harness a confident call with invented
arguments is worse than no call, so it is reported as a regression.

### The owner's suite (dsbench v2, pi harness, k = 5): the target hit, and a new failure mode

89/115 → 87/115 passing runs; 18/23 → 19/23 problems by majority (2 gains, 1 loss, p = 1.0). With 23
problems the suite has almost no statistical power, so the per-problem picture matters more:

| Problem | Base | Adapter | Note |
|---|---:|---:|---|
| `da_utc_peak_hour` | 0/5 | 4/5 | the UTC / timezone convention: exactly Target A's gap |
| `da_delay_streak` | 2/5 | 3/5 | |
| `de_route_leaderboard` | 4/5 | 5/5 | |
| `da_worst_dep_hour` | 3/5 | 0/5 | timeouts and wrong answers |
| `de_hub_daily` | 5/5 | 3/5 | a timeout and a wrong answer |
| `da_ts_delay`, `da_weekend_delay`, `de_recovery_leaderboard` | 1, 4, 5 | 0, 3, 4 | |

The adapter learned the thing Target A taught: the problem the baseline failed five times out of
five on a ClickHouse time-zone convention now passes four times. Across all runs it is also wrong
less often (19 wrong answers vs 26) and makes fewer tool errors (22 vs 56 over its completed runs,
in 6.4 turns per run vs 7.4). What cancels that out is a failure the base never shows: 9 of its 115
runs hit pi's 600-second limit (base: 0). The server log shows single replies running to 16,384
tokens, the same greedy repetition loop IFEval shows (24 replies at the 4096-token cap vs 9). In an
agent loop nothing caps a reply, so one loop costs the whole run. Without the loops the adapter
could have reached up to 96/115.

## The think-leak: the adapter reasons although thinking is off

With thinking off, the server's prompt ends in an empty block, `<|im_start|>assistant\n<think>\n\n
</think>\n\n`, and the base model answers directly (0 of 164 HumanEval+ replies start with
`<think>`). The adapter opens a second block on its own: 54 of 163 HumanEval+ replies begin with
`<think>`, and 12 of them never close it, running into the 2048-token limit with no code written.
That, not worse code, is the HumanEval+ regression: on the 109 items where neither state leaked,
the delta is −0.9 pp (1 gain, 2 losses, p = 1.0).

The cause is in the training text. Of the 30,285 assistant turns in the tokenized training blocks,
1,762 begin with a real reasoning block and 25,928 with the template's empty one, and in all 27,690
of them `<think>` is a trained token. (The other 2,595 are earlier turns of multi-turn
conversations, which the template renders without a block.) The reasoning comes from three pools
of the mixture:

| Pool | Assistant turns | Real reasoning | Empty block | No block |
|---|---:|---:|---:|---:|
| targetA | 3,392 | 1,403 | 1,989 | 0 |
| datamind | 640 | 640 | 0 | 0 |
| swe_swiss | 222 | 222 | 0 | 0 |
| all other pools | 26,085 | 0 | 0 | 26,085 |

(The mixture file holds 2,265 reasoning turns; the chat template drops reasoning from turns that
precede the last user message, leaving 1,762 in the tokenized text.) `render.py` bakes a teacher's
reasoning into the assistant turn by design ("the model keeps its thinking channel"). But the model
is served, and was evaluated, with thinking off, where the empty block is part of the *prompt*, not
something the model generates. Training on the `<think>` tokens taught "an answer starts by
reasoning" on code-like prompts, and the adapter now does that after the template's empty block too.

The leak follows the kind of prompt. Asked to write a program from a problem statement, the
adapter reasons first: HumanEval+ 54 of 163, LiveCodeBench 246 of 342 (base: 0 and 33). On prose it
rarely does (IFEval 15 of 541, MMLU-Pro 12 of 1400, all closed). DS-1000 shows none at all (0 of
1000): its system prompt demands bare code between `<code>` tags, and the adapter's replies there
are shorter than the base's (46 vs 103 tokens on average).

**Addendum (2026-09-29).** The training tokenizer did not split `<think>` and `</think>` the way the
server does. transformers' GGUF converter left them as BPE pieces (`<th` `ink` `>`), while
llama.cpp serves each as one token. So the empty block the adapter was served never appeared in its
training text in that form. That fits the leak without proving it; see
`20260929-thinking-rendering-packing.md`.

## MTP acceptance: passes

The Qwen3.6 MTP head (exported with the fork converter's `--mtp` mode, Q8_0) drafted for the same
production binary, with one slot, greedy, and its default 3 draft tokens. It ran on 200 fixed battery
prompts (the first 50 of IFEval, HumanEval+, DS-1000 and MMLU-Pro), up to 512 tokens each, once per
state (`mtp.py`, `battery_mtp.sh`).

| | Base (scale 0) | Adapter (scale 1) |
|---|---:|---:|
| Draft tokens proposed | 50,523 | 41,522 |
| Accepted | 33,449 | 28,801 |
| Acceptance | 66.2% | 69.4% |
| Decode speed, mean per request | 88.9 tok/s | 75.8 tok/s |

Acceptance doesn't drop; it rises by 3.2 points, so the ADR's criterion (a drop of at most 5
points) passes. The MTP layer is frozen and knows nothing of the adapter, but it drafts from the
adapted trunk's hidden state, and on the adapter's own output it is right more often. Each state is
measured on the text it generates, which is what a user of either would get. The adapter decodes
14.6% slower. That is the cost of applying the LoRA at runtime (at scale 0 the server skips it), not
an acceptance effect, and it sits just under the 15% that ADR-001's Gate 1 set as the limit for
serving an adapter at runtime. A merged adapter would not pay it.

## ADR step 4: the adapter at half scale

ADR-001's remedy for a regression begins: "halve the adapter scale at load and re-evaluate". The
same server ran the seven public benchmarks once more at LoRA scale 0.5, paired with the same base
pass (state `adapter_half`). The owner's suite and MTP were not rerun. GPQA, gated until 2026-09-28,
ran all three states in a window of its own that day.

| Benchmark | n | Base | Scale 1.0 | Scale 0.5 | Δ at 0.5, pp (95% CI) | +/− | p | Verdict at 0.5 |
|---|---:|---:|---:|---:|---|---:|---:|---|
| DS-1000 | 1000 | 47.1 | 47.8 | 49.9 | +2.8 (+0.1, +5.4) | +103/−75 | 0.043 | gain |
| BIRD | 1532 | 60.1 | 51.0 | 57.4 | −2.6 (−4.3, −0.8) | +75/−115 | 0.005 | regression |
| BFCL AST | 1000 | 88.5 | 91.5 | 94.3 | +5.8 (+4.4, +6.7) | +66/−8 | 2e-12 | gain |
| IFEval | 541 | 81.7 | 77.1 | 81.9 | +0.2 (−2.8, +3.2) | +32/−31 | 1.0 | pass (unresolved) |
| MMLU-Pro | 1400 | 80.8 | 77.7 | 80.8 | 0.0 (−1.5, +1.5) | +55/−55 | 1.0 | pass (unresolved) |
| GPQA | 198 | 59.1 | 67.7 | 64.1 | +5.1 (−0.7, +9.7) | +19/−9 | 0.087 | pass (confirmed) |
| LiveCodeBench | 342 | 37.4 | 30.1 | 33.6 | −3.8 (−7.1, 0.0) | +13/−26 | 0.053 | fail (n.s.) |
| HumanEval+ | 163 | 90.8 | 85.9 | 92.0 | +1.2 (−0.8, +1.2) | +2/−0 | 0.5 | pass (confirmed) |
| BFCL irrelevance | 240 | 89.2 | 67.1 | 69.6 | −19.6 (−20.4, −16.0) | +1/−48 | 2e-13 | regression (no ADR limit) |

("Unresolved" passes have a point estimate inside the limit but an interval that reaches past it.)

At half scale most regressions go away and the gains don't:

- **Three of the five general benchmarks return to the base.** IFEval 81.9 (base 81.7), MMLU-Pro
  80.8 (80.8), HumanEval+ 92.0 (90.8). The think-leak is gone (no reply opens a `<think>` block on
  HumanEval+, IFEval or MMLU-Pro, and 1 of 342 on LiveCodeBench), and so is most of the brevity:
  MMLU-Pro's median reasoning is 628 tokens (base 728, full scale 386), GPQA's 2,074 (base 2,118,
  full scale 1,469), and IFEval's median reply is 803 characters (base 850, full scale 694).
- **GPQA stays up**, +5.1 (p = 0.087), for the full-scale reason but more weakly: the limit cuts
  53 replies against the base's 68, and 18 of the 19 gains are questions the base ran out of
  tokens on. Where both states answered, the half-scale adapter is right 88.6% of the time and the
  base 91.1% (1 gain, 4 losses, p = 0.38).
- **LiveCodeBench does not.** 33.6 against 37.4 is −3.8 points, past the 2-point limit (p = 0.053,
  so the ADR's size rule fails it, as it did HumanEval+ at full scale). Without the leak, the
  half-scale adapter writes code about as often as the base (170 replies without code vs 175),
  but more of it is wrong (57 wrong answers vs 38; medium problems 42.3 → 32.7). That is the one
  regression here that is not about format.
- **The tool-call gain grows**, to +5.8 on BFCL AST (parallel calls 81.0 → 94.0).
- **DS-1000 becomes a gain**, +2.8. With BFCL AST that makes the two target gains the ADR asks for,
  but this one is marginal. Its p is 0.043, the A/A pass alone moved the base by +0.8, and one p
  just under 0.05 among this report's nineteen paired tests is weak evidence on its own.
- **Two defects survive:** BIRD still regresses (−2.6; 77 SQLite errors vs the base's 47), and the
  adapter still calls tools that don't fit (irrelevance −19.6, barely better than full scale).

The dose response separates the failures. The think-leak and the brevity grow with the adapter's
strength, and at half strength neither shows any more; of the forgetting, only LiveCodeBench's
remains. Calling a tool whenever tools are offered, and writing MySQL date functions for SQLite,
are what the data taught rather than how strongly it taught them: at half strength they shrink
(the tool calls barely) but stay.

## Gate-2 decision: not passed

**At full scale the adapter fails both halves of the criterion.** ADR-001 needs significant gains
on at least two of the four target benchmarks, and only BFCL AST gains (+3.0 pp, p = 0.004).
DS-1000 and the owner's suite are flat, and BIRD goes backwards (−9.1 pp). Four of the five general
benchmarks fail their limits: IFEval −4.6 (p = 0.014) and MMLU-Pro −3.1 (p = 0.0008) against 1
point, and LiveCodeBench −7.3 (p = 0.0006) and HumanEval+ −4.9 (p = 0.077) against 2 points. The
ADR bounds the size of a regression and does not ask for significance, so HumanEval+ fails on its
point estimate, all of which comes from the think-leak. GPQA passes, +8.6 (p = 0.012), on the
token budget rather than on better reasoning: 28 of its 29 gains are questions where the base ran
out of tokens, and where both states answered, the adapter is worse (84.1% vs 90.5%). MTP
acceptance passes: it rises 3.2 points.

**At half scale (step 4) it still fails, more narrowly.** The target criterion is met on paper by
BFCL AST (+5.8) and DS-1000 (+2.8, p = 0.043), but LiveCodeBench regresses 3.8 points against its
2-point limit. Even without that, the half-scale adapter should not close Gate 2:

- its second target gain is marginal;
- BIRD, a target benchmark, still regresses (−2.6, p = 0.005);
- it still calls tools where none fit (−19.6). The criteria don't bound this, but it matters most
  in the agent harnesses this tune is for;
- two measurements are missing at half scale: the owner's suite and MTP were not rerun.

**Next, per the ADR:** step 4 goes on to "raise replay to 40% and retrain", and the table below
lists what else the retrain needs. Step 4's last clause asks for the responsible data slice to be
found by ablation. The data analysis above mostly answers that without a training run: the think
blocks come from three pools, the always-call habit from all 896 tool rows, and the SQLite errors
from Gretel.

## What the retrain has to change

ADR-001's remedy for a regression is to halve the adapter scale, then raise replay to 40% and
retrain. Replay addresses forgetting, which is what IFEval, MMLU-Pro and LiveCodeBench look like.
The other three failures are specific defects in the training data, and more replay would dilute
them without removing them:

| Failure | Cause, from the evidence above | Change for the retrain |
|---|---|---|
| Think-leak: HumanEval+ −4.9 (54/163 replies open a second `<think>`), LiveCodeBench −7.3 (246/342, 169 never closed) | `<think>` is a trained token in 27,690 assistant turns, 1,762 of them with real reasoning (mostly Target A) | Render every row in the format it is served in. With thinking off, the empty block is part of the prompt and gets no label; the teacher's reasoning is dropped, or those rows are trained with thinking on and served with thinking on |
| Calls where no tool fits: BFCL irrelevance −22.1 | all 896 rows that offer tools open with a tool call; none declines | Add decline rows: an offered tool list that doesn't fit the question, answered in plain text (generated, not BFCL items) |
| BIRD −9.1: `YEAR()`, invented columns | the SQL pool is Gretel synthetic, and one in five of its training answers fails in SQLite against its own schema (MySQL date functions, missing columns and tables); B.4's BIRD/Spider train rows were never added | Cut Gretel, or keep only rows that execute against their own schema; add BIRD and Spider train in SQLite with the full schema in the prompt, as B.4 planned; name the dialect in every SQL prompt |
| IFEval −4.6, MMLU-Pro −3.1: replies about half as long, repetition loops; LiveCodeBench −3.8 even at half scale, with more wrong code; GPQA +8.6 only because shorter reasoning fits the 4096-token budget (where both states answered, the adapter is worse, 84.1% vs 90.5%) | forgetting, and a length prior: the trained assistant turns are short (median 94 tokens; per pool, a median of 241 to 1,180 characters, except SWE), and the replay is no exception (Tulu 3: 513 characters) | ADR step 4, replay to 40%, with replay chosen for length as well as topic: instruction-following rows with explicit constraints, and long step-by-step answers; pi and any agent harness cap each reply's tokens, so one loop can't eat a run |

**Check a checkpoint before the next full battery.** Each of the four failures moves far past its
A/A noise on a few hundred items, so a one-hour mini-battery catches them all: IFEval, BFCL
irrelevance, BIRD's SQLite error count and HumanEval+'s think-leak rate. The Gate-2 loss eval could
not: its forgetting control, tulu3 rows neither adapter trained on, improved from base to Gate 2
(assistant loss 2.05 → 0.94) while IFEval fell 4.6 points. Held-out loss on text from the
mixture's own sources measures how well the model fits that style, not whether it still follows
instructions.

## Ops notes

- **8 slots is the throughput optimum on this box.** 104 tok/s at 8 slots, 40 to 80 at 16. Vulkan's
  mat-vec kernels serve batches of up to 8 tokens (`mul_mat_vec_max_cols = 8`); past that, MoE
  decode falls to the general matmul path.
- **Never mix LoRA states concurrently.** llama-server batches a slot only with slots whose LoRA
  config matches the first busy slot's; a request in the minority state waits until the majority
  drains. The battery runs one state per pass.
- **`--lora-init-without-apply` does not make "no state" mean base** on this build: the global
  scale stays 1.0. The window posts scale 0 at start.
- **Podman on Fedora needs `:z` on bind mounts** (SELinux), and llama-server's slot `erase` needs
  `--slot-save-path`.
- **Don't stop the OCR unit while it is still loading.** Between two windows the first one's exit
  trap restarted OCR and the next window stopped it 16 seconds later. systemd's stop timed out and
  killed only the `toolbox run` wrapper, and the llama-server inside the container survived, stuck,
  ignoring SIGTERM and holding GPU memory. It was killed by hand, and `stop_ocr` in both launchers
  now waits for OCR's `/health` before stopping it and kills any OCR server that outlives the stop.
- **`/tmp` on dashi is tmpfs, and a reboot empties it.** The GPQA window's first server start
  failed: `--slot-save-path /tmp/battery-slots` was gone after the Sep 25 reboot, and llama-server
  refuses a missing directory. The launcher now creates it. The window also restores OCR only if
  OCR was running when the window began; it had stayed off since that reboot.
- **Budget the scorer for the whole battery.** The battery took 28 hours of box time (Sep 24 12:29
  to Sep 25 16:43: 19.8 h main window with the pi suite, 25 min MTP, 8 h half scale); the
  autoscore loop had been started with a 16-hour deadline, and a successor had to be chained in.
  MMLU-Pro alone is ~3 h per state at this budget (4096 tokens, zero-shot CoT).

## Addendum, 2026-10-01: every tool call trained without its arguments

This was found while building Target C's agentic pilot for the retrain, and confirmed on the tokens
Gate 2 trained on. The training run is in `~/benchlab/runs/2026-09-23-qwen36-gate2-train/`, under
`data_tokenized_gate2`.

- **The mechanism.** The mixture records tool calls in the OpenAI format, with `arguments` as a JSON
  string (all 6,096 of its calls). The model's template renders a call's parameters only when
  `arguments` is a mapping. Given a string, it renders `<function=run_sql>\n</function>`, a call
  with no arguments. llama-server parses the string into an object before it applies the
  template, so in serving the model sees its parameters. The training tokenizer, transformers'
  `apply_chat_template`, doesn't.
- **What was trained.** Decoding all 4,423 training blocks:
  - 6,061 real tool calls were empty: `add_and_execute_jupyter_code_cell` 2,561, `run_sql` 1,851,
    `run_python` 760, `final_answer` 557 and `finish` 332;
  - the 888 calls that carry parameters are all the template's own format example,
    `example_function_name`;
  - the tool responses were rendered: 30 of 4,724 are empty, as in the source data.

  So the adapter's tool-using rows (jupyter-agent and Target C) taught it to emit calls with no
  arguments, after reasoning that plans them.
- **What it may explain.** Not measured separately. The adapter still filled its arguments in on
  BFCL AST (+3.0) and on the owner's suite, so the base's own behaviour held. The irrelevance
  regression (−22.1: calls where no tool fits) has a second candidate cause beside "every row
  that offers tools opens with a call": rows that teach calling as a reflex, with nothing in it.
- **The fix for the retrain** (ADR-004 Revision 2, "Rendering and packing"):
  `tokenize_masked.thinking_record` parses the arguments as the server does, and rejects a row
  whose arguments aren't a JSON object. The legacy path that built Gate 1 and Gate 2 is left as it
  was, so those runs stay reproducible.

## Addendum, 2026-10-09: no pass reused another state's prompt cache

llama-server matches cached prompts by tokens alone, whichever LoRA scale computed them, and its
RAM copy of prompts loads into a slot without checking the scale. In the Revision 2 and 2.1 probe
windows, two probe problems' adapter runs started from the base's prompt KV that way
(`patches/README.md`, "The prompt cache across states"). This battery's windows, read from their
server logs the same day, had none of it:
- **The paired passes:**
  - Each request names its scale, so a pass's first request on each slot drops that slot's
    cache.
  - The RAM copy holds only about 26 BIRD or 60 other prompts, and drops the oldest first. The
    later reuse within a pass came from the pass's own items. BIRD's early loads reused 2,300 to
    3,180 schema tokens, and the base's last items are another database. DS-1000's one early load
    reused 73 tokens, more than the base's last items share with it.
  - HumanEval+ and IFEval reused nothing at all.
- **The other steps:**
  - `hold:dsbench` (pi): no problem's first run in either step reused anything.
  - GPQA (base, adapter, half scale): no request loaded from the RAM copy.
  - The half-scale window served one state.
  - The MTP window: no scale-1 request reused its own scale-0 prompt.

No number or conclusion above changes. The scripts and their output are on the box, in
`~/benchlab/scratch/gate2-cache-audit-20261009/` (its `README.md`).
