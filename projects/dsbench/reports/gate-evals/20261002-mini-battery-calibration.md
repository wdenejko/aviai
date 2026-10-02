# The mini-battery's calibration: the base twice, with thinking on

ADR-004 Revision 2, action item 8. Run 2026-10-02 on dashi in two battery windows
(`patches/battery_window.sh`; box clock, about 2 hours behind CEST):
- 10:25-19:25, `MAX_HOURS=9`: every pass but the last 34 items of BIRD's A/A pass, which the time
  limit stopped;
- 19:26-19:37, `MAX_HOURS=3`, armed to start as the first ended: those 34 items.

The base answered the mini-battery's items twice with thinking on: `base`, then `base_rep` with
other seeds, the A/A pass. Outputs:
- the numbers are in `20261002-mini-battery-calibration.json`:
  `patches/mini-battery/calibration.py`'s, and `mini summary` of the A/A pass against the base;
- the raw outputs are in `~/benchlab/runs/2026-10-02-qwen36-mini-battery-base/`.

## Summary

- **A checkpoint's gate takes about 5 hours, one window.** The base's passes took 4.5 and 4.7
  hours a state: HumanEval+ 122-126 minutes, IFEval 74-79, BIRD 42-46, BFCL 28-33. Gate 2
  served its adapter 14.6% slower than the base.
- **Thinking moves the base's own scores, both ways.** Against Gate 2's thinking-off base on the
  same items: IFEval +9.0 points, BIRD +9.3, BFCL irrelevance −10.8, HumanEval+ −6.8, each with
  p ≤ 0.043. Gate 2's numbers are not the retrain's baseline; this calibration is.
- **The base loops.** A reply that never closes its reasoning runs to the 12,288-token cap and
  answers nothing: HumanEval+ 18 and 16 of 163, BFCL 7 and 10 of 120, IFEval 5 and 5 of 200,
  BIRD 0 and 1 of 150. That is where thinking loses: 15 of HumanEval+'s 18 losses and 7 of
  BFCL's 14 never closed. HumanEval+'s replies that closed passed 94.5% of the time.
- **The noise.** Two draws of the base disagree on 8.5% of IFEval's items, 11.7% of BFCL's,
  16.0% of HumanEval+'s and 9.3% of BIRD's. To flag at 80% power, a checkpoint's passes must
  move by about 5.8, 8.7, 8.8 and 7.0 points. The A/A pass raised no flag.
- **The subsets would have caught Gate 2.** The same checks on Gate 2's own passes (thinking off)
  flag its adapter on every benchmark at full scale, each through the habit the subset was sized
  for. At half scale only BFCL flags, as Gate 2's dose check found. Pass rates alone would have
  missed IFEval and HumanEval+.
- **The sizes and the budget stay.** At 8,192 tokens, a fifth of HumanEval+'s closed replies
  would no longer close, so the gate would stop measuring what pi serves. Most of what a
  smaller budget saves is loops.
- **A finding for Revision 2's data.** The same fifth of the base's closed HumanEval+ replies
  runs past 8,192 tokens, the training block. If the code pools' replies run as long, their rows
  keep only the shorter ones. Shorter code reasoning is what the brevity check would catch.
- **The full battery with thinking on costs several windows a state.** At this calibration's
  cost per item, its four benchmarks measured here would take about 18 hours a state at full
  size. The other four were not measured with thinking on.

## What ran

- **The items** are the mini-battery's subsets, as `mini prepare` drew them (seed 20261002):
  - IFEval: 200 of 541;
  - BFCL: 120 of its 240 irrelevance items;
  - BIRD: 150 of 1,534, by difficulty (91 simple, 45 moderate, 14 challenging);
  - HumanEval+: all 164. HumanEval/32 can't be measured, as in Gate 2: the dataset's own
    reference fails its test.
- **The server** was Gate 2's battery server: the I-Mini quantization on the Vulkan build, 8 slots
  sharing a 196,608-token cache, Gate 2's adapter loaded at scale 0. Gate 2's parity check showed
  that scale 0 serves the base.
- **Sampling.** Thinking on, with Qwen's settings for it: temperature 0.6, top-p 0.95, top-k 20,
  min-p 0. A reply gets at most 12,288 tokens, pi's reply cap. An item draws the same seed in
  every state; `base_rep` adds 1 to it.
- **The order:** BFCL, IFEval, HumanEval+, BIRD, each `base` then `base_rep`. `battery.autoscore`
  scored each pass as it completed, with Gate 2's scorers.
- **The code** was the box's synced copy of `battery/`. Its `generate`, `score` and `items` predate
  PR #33, which moved a function between modules and changed nothing they do.

## Cost

| Benchmark | Pass | Minutes | Tokens | Median | p90 | Max | Never closed |
|---|---|---:|---:|---:|---:|---:|---:|
| BFCL irrelevance | base | 27.8 | 197,514 | 768 | 2,452 | 12,288 | 7 of 120 |
| | base_rep | 33.4 | 233,116 | 836 | 3,003 | 12,288 | 10 of 120 |
| IFEval | base | 73.8 | 583,362 | 2,392 | 5,205 | 12,288 | 5 of 200 |
| | base_rep | 79.2 | 618,476 | 2,374 | 5,921 | 12,288 | 5 of 200 |
| HumanEval+ | base | 126.0 | 867,801 | 3,518 | 12,288 | 12,288 | 18 of 163 |
| | base_rep | 122.2 | 864,534 | 3,466 | 12,057 | 12,288 | 16 of 163 |
| BIRD | base | 42.2 | 294,260 | 1,759 | 3,439 | 5,802 | 0 of 150 |
| | base_rep | 45.8 | 311,195 | 1,836 | 3,326 | 12,288 | 1 of 150 |

- **Tokens** are the replies' (reasoning and answer), over the measurable items. The server wrote
  113-132 of them a second across its 8 slots, against 104 in Gate 2's thinking-off battery:
  longer replies keep the slots full.
- **A state's four passes took 4.5 and 4.7 hours** (270 minutes for `base`, 281 for `base_rep`).
  A checkpoint's would take about 5.2: Gate 2 measured its adapter, applied at runtime, 14.6%
  slower than the base.
- **HumanEval+ takes nearly half of it.** Its replies are the longest (3,500 tokens at the
  median), and a tenth of them loop to the cap.

## The base loops

A reply that never closes its reasoning runs to the 12,288-token cap and answers nothing.
- **How often:** HumanEval+ 18 and 16 of 163, BFCL 7 and 10 of 120, IFEval 5 and 5 of 200,
  BIRD 0 and 1 of 150.
- **It follows the item, partly.** In both passes: 6 HumanEval+ items, 4 BFCL items, 2 IFEval
  items.
- **What the loops do,** read in each reply's last 3,000 characters:
  - **BFCL: the answer is settled, and the reply keeps writing it.** In all 17, the reply has
    decided not to call: it declines, or answers from what it knows. Then it checks that decision
    again and again (13), or writes its final answer over and over (4), and never closes.
  - **IFEval: the reply is stuck on a constraint on its wording,** such as a letter to avoid,
    capitals, or a rhyme. It tries one candidate a line, redrafts the same lines, or repeats one
    line verbatim. Where each line differs, counting repeated sentences misses the loop.
  - **HumanEval+: the reply keeps checking its program, and how to present it.** It traces
    examples by hand, finds nothing wrong, and checks once more, or weighs again whether the
    script should carry tests or its docstring. In most, nearly every line is new.
  - **BIRD's one: a candidate query, rejected over and over,** the same line 42 times. On the
    other draw, the same item closed in under 3,000 tokens.
- **They cost a pass more than their share:** 43.5% and 52.7% of BFCL's tokens, 25.5% and 22.7%
  of HumanEval+'s.
- **BFCL's scorer passes them.** Irrelevance passes when no call is decoded, and a reply that
  never closes decodes none. The mini-battery's passes now need closed reasoning
  (`mini.conditions`), found on this calibration's first pass. The other scorers already failed
  their loops.

## Thinking on, against Gate 2's thinking-off base

The same items, Gate 2's base (greedy, thinking off) against this `base`, paired:

| Benchmark | n | Thinking off | Thinking on | Gains / losses | p |
|---|---:|---:|---:|---|---:|
| IFEval | 200 | 79.0 | 88.0 | 24 / 6 | 0.0014 |
| BFCL irrelevance | 120 | 92.5 | 81.7 | 1 / 14 | 0.001 |
| HumanEval+ | 163 | 90.8 | 84.0 | 7 / 18 | 0.043 |
| BIRD | 150 | 60.7 | 70.0 | 17 / 3 | 0.0026 |

- **Thinking moves the base both ways,** so Gate 2's numbers are not the retrain's baseline. The
  checkpoint gate compares with this pass.
- **BFCL's 14 losses:** 7 replies that never closed, and 7 calls where none fits. Of the 9
  items the base calls a tool on with thinking off, it calls one on 8 with thinking on. It declines
  the ninth, the one gain.
- **HumanEval+'s 18 losses:** 15 replies that never closed, and 3 closed ones whose program
  failed. The replies that closed passed 137 of 145 times (94.5%).
- **BIRD gains on simple and moderate questions** (61 to 69 of 91, 19 to 26 of 45; challenging
  11 to 10 of 14). SQL that SQLite can't run stays at 4 of 150.

## Noise: the A/A pass

`base_rep` against `base`, with the mini-battery's own comparison (`mini summary --state base_rep`):

| Benchmark | Discordant items | Smallest detectable change | Passes, base / base_rep | Reasoning length ratio |
|---|---:|---:|---|---|
| IFEval | 17 of 200 (8.5%) | 5.8 points | 88.0 / 87.5 (p = 1) | 1.016 (92 shorter, 100 longer; p = 0.61) |
| BFCL irrelevance | 14 of 120 (11.7%) | 8.7 points | 81.7 / 78.3 (p = 0.42) | 1.095 (47 / 60; p = 0.25) |
| HumanEval+ | 26 of 163 (16.0%) | 8.8 points | 84.0 / 86.5 (p = 0.56) | 0.990 (66 / 69; p = 0.86) |
| BIRD | 14 of 150 (9.3%) | 7.0 points | 70.0 / 68.7 (p = 0.79) | 1.008 (66 / 83; p = 0.19) |

- **No check flagged the A/A pass.** Its passes, its loops, BIRD's SQLite errors and its
  reasoning length all stayed within the test's noise.
- **The smallest detectable change** is at 80% power for an exact McNemar test at 5%, from the
  discordance by the normal approximation. A checkpoint can't be less discordant with the base
  than another draw of the base, so the figure is optimistic.
- **Reasoning length moved by 1% to 10% between two draws of the base** (ratios 0.990 to 1.095).
  The brevity check's floor, 0.8, sits well outside that.
- **BIRD's SQLite errors held at 4 of 150 in both draws,** 3 items erring in one draw and 3 in
  the other.

## Gate 2 in retrospect: would these subsets have caught it?

The mini-battery's checks, run on Gate 2's adapter against its base over the same subsets
(`calibration.py`'s `gate2_flags`). Gate 2 ran greedy with thinking off, so there is no reasoning
to compare in length and none to leave unclosed; its think-leak opened a reasoning block inside
the reply, which is the `second_think` check.

| Benchmark | Passes: base, adapter (better / worse, p) | Adapter's flags | At scale 0.5 |
|---|---|---|---|
| IFEval | 79.0, 76.5 (14 / 19, p = 0.49) | truncated: 2 to 11 items (p = 0.022); second think: 0 to 8 (p = 0.008) | no flag |
| BFCL irrelevance | 92.5, 62.5 (0 / 36, p < 1e-10) | passes; second think: 0 to 25 | passes: 92.5 to 69.2 (0 / 28) |
| BIRD | 60.7, 51.3 (7 / 21, p = 0.013) | passes; SQLite errors: 4 to 18 (p = 0.003) | no flag (errors 4 to 10, p = 0.11) |
| HumanEval+ | 90.8, 85.9 (4 / 12, p = 0.077) | truncated: 0 to 16 of 163; second think: 0 to 54 (p < 1e-15) | no flag |

- **At full scale, every benchmark flags,** each through the habit the subset was sized for.
- **At half scale only BFCL flags,** which is what Gate 2's dose check found: there the leak was
  gone and BFCL still failed.
- **The pass rates alone would have missed IFEval and HumanEval+.** IFEval's 2.5 points on this
  subset are under its 5.8-point floor. The leak caught both here. A retrain without the leak but
  with Gate 2's diffuse IFEval loss would rest on the brevity check, which this retrospective
  can't test: Gate 2's replies had no reasoning. The full battery stays the final check.
- **Greedy decoding left little noise:** two of Gate 2's base passes differed on 5 of 163
  HumanEval+ items. With thinking on, two draws disagree on 8.5-16% of items, so the same habits
  would sit closer to the line. BFCL's 36 to 0 would not move far. Nor would BIRD's 17 new
  SQLite errors against 3 gone: two draws of the base differ by 3 and 3.

## The budget

What a smaller budget than 12,288 would cut: closed replies that ran past it (they would no
longer close), and the tokens saved.

| Benchmark | Pass | 4,096 | 6,144 | 8,192 | 10,240 |
|---|---|---|---|---|---|
| BFCL irrelevance | base | 3 cut, 32.3% saved | 2, 22.5% | 0, 14.5% | 0, 7.3% |
| | base_rep | 2, 38.3% | 1, 28.6% | 1, 19.0% | 1, 9.3% |
| IFEval | base | 27, 13.6% | 7, 6.5% | 1, 3.8% | 0, 1.8% |
| | base_rep | 30, 17.9% | 13, 8.6% | 4, 4.6% | 2, 2.0% |
| HumanEval+ | base | 48, 41.9% | 37, 27.8% | 26, 16.2% | 18, 6.6% |
| | base_rep | 53, 42.0% | 41, 27.1% | 31, 14.4% | 15, 5.4% |
| BIRD | base | 4, 0.9% | 0, 0 | 0, 0 | 0, 0 |
| | base_rep | 5, 4.3% | 1, 2.0% | 0, 1.3% | 0, 0.7% |

- **The budget stays at 12,288, pi's reply cap.** Below it, HumanEval+ loses closed replies first:
  26 and 31 of its 145 and 147 at 8,192, 18 and 15 at 10,240. Each would become a failure that
  pi, serving 12,288, doesn't produce.
- **What a smaller budget saves is mostly loops:** BFCL's 14.5-19.0% at 8,192 is almost all
  replies that never closed.
- **A finding for Revision 2's data.** Training rows hold 8,192 tokens, prompt included, and the
  single-turn pools are generated to fit (`--block 8192`). A fifth of the base's closed HumanEval+
  replies (26 of 145, 31 of 147) run past that. If OpenCoder's and SWE-Swiss's replies run as
  long, the code rows keep only the shorter ones, which could teach shorter code reasoning. The
  generation window will count how many it cuts, and the brevity check compares a checkpoint's
  reasoning length on HumanEval+ with the base's.

## What it means for the checkpoints and the full battery

- **One checkpoint, one window.** About 5 hours for its four passes. At 10M tokens the recipe
  saves a checkpoint every 100 steps, about a dozen in all, so gating every one would take some
  60 hours (batch 1, so a step is a block). Proposed: the final checkpoint first; if it flags,
  earlier ones.
- **The full battery, re-baselined with thinking on** (action item 8's last point), at this
  calibration's minutes per item:

  | Benchmark | Items | Hours a state |
  |---|---:|---:|
  | IFEval | 541 | 3.4 |
  | BFCL | 1,240 | 5.3, if its call categories cost what irrelevance does |
  | BIRD | 1,534 | 7.5 |
  | HumanEval+ | 164 | 2.1 |

  That is about 18 hours a state before MMLU-Pro (1,400 items), GPQA (198), LiveCodeBench (342)
  and DS-1000 (1,000), which this calibration didn't measure with thinking on. With thinking off
  and 4,096 tokens, Gate 2's base already hit its limit on 176 of LiveCodeBench's 342 items. So
  the re-baseline is several windows a state, for the base and for each candidate: decision 9
  in ADR-004.

## What changed with this calibration

- **A pass needs closed reasoning** (`mini.conditions`). Found on BFCL's first pass: its scorer
  passed 7 replies that answered nothing. Counted as passes, a checkpoint that loops more would
  score better.
- **A pass is compared on the items it answered** (`mini.bench_summary`). A pass stopped by the
  window's time limit has score rows for the rest (`no_generation`), which say nothing about the
  model.
- **The summary reports what each pass cost** (`mini.cost`): tokens per item and in total, and the
  summed request latency.
- **Only the subset's own unmeasurable items are listed.** The gold scores cover the whole battery,
  so BIRD listed two items that aren't in its subset.
- **`patches/mini-battery/calibration.py`** reads the run for this report, across both windows.

## Limits

- **One base, two draws.** The noise above is one A/A pass per benchmark.
- **The retrospective ran thinking off and greedy,** as Gate 2 did. It shows the subsets are large
  enough for Gate 2's habits, not that a thinking-on checkpoint with the same habits flags as
  surely.
- **The full battery's estimate scales minutes per item.** Eight slots stay fuller on a longer
  pass, so it is rough, and BFCL's call categories are assumed to cost what irrelevance does.
