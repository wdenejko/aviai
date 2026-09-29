# Decontamination against the acceptance battery: a gate before training

**Date:** 2026-09-29 · raw numbers: `20260929-battery-decontamination.json` · code:
`src/dsbench/sftgen/decontaminate.py` (rule 4), `src/dsbench/battery/contamination.py`
(`build_index`, `shared_units`) · tests: `tests/test_decontaminate_battery.py`

**Status: built (ADR-004 Revision 2, action item 7).**
- **The gate.** `decontaminate.py --battery-items DIR` rejects a training row that reproduces at
  least a fifth of a battery item, or a short item whole. `assemble.py` takes the same flag.
- **Checked against the Gate-2 mixture,** it finds what the after-the-fact check found: the same
  44 items, and it rejects the 2 rows behind the one item over the line.
- **The new reach.** It also sees what that check could not: 386 short BFCL, IFEval and BIRD
  requests, and BFCL's function schemas.

## Why

The Gate-2 mixture was decontaminated against dsbench only. Its exposure to the battery was
measured after training, by `battery/contamination.py`; see the battery report's "Contamination"
note. The retrain draws new prompts for every pool, so the check moves before training and becomes
a gate.

## The rule

It uses the same item text and the same 13-grams as the after-the-fact check: the problem,
question or reference solution, never a benchmark's shared boilerplate.

- **A row is rejected** when it covers at least 20% of an item's distinct 13-grams, the line that
  check calls "strong". A single 13-gram is not enough, unlike dsbench's rule 1. dsbench's prompts
  are distinctive aviation text, while the battery is generic code and prose: all 44
  single-13-gram overlaps in Gate 2 were generic (counting sequences, textbook Fibonacci, the
  definition of a subsequence).
- **Rows below the line are kept** and listed in the report's audit, item by item, so every
  overlap stays visible.
- **The whole row is scanned:** prompt, reasoning, answer, tool-call arguments and offered tool
  schemas. The reasoning (`reasoning_content`, where Revision 2's rows keep it) and the schemas
  were not scanned before.

It adds two things the after-the-fact check did not need:

1. **Short items.** An item shorter than 13 tokens has no 13-gram, so no 13-gram index can see
   it. That covers 397 of BFCL's 1,240 requests: exactly the requests a generated tool row could
   reproduce, such as "Calculate the circumference of a circle with radius 3".

   | Item length (tokens) | BFCL | IFEval | BIRD | Handling |
   |---|---:|---:|---:|---|
   | 3-5 | 14 | | | too generic to match ("Who discovered radium?"); counted as unchecked |
   | 6-7 | 68 | | | matched whole |
   | 8-12 | 315 | 2 | 1 | matched whole |

2. **BFCL's function schemas,** as units of their own. Revision 2's tool rows must not reuse them.
   Both sides are dumped as JSON with sorted keys, so key order can't hide a copy.

## Validation

| Mixture | Rows | Rejected | Kept with an overlap below the line | Time |
|---|---:|---:|---|---:|
| Gate 2 | 23,640 | 2 | 55 rows, touching 43 items | 5.4 s |
| Gate 1 pilot | 2,095 | 0 | 10 rows, touching 19 items | 1.0 s |
| Reasoning pilot (thinking on, the base's own replies) | 199 | 0 | 2 rows, touching 16 items | <1 s |

- **Agreement.** The items touched in the Gate 1 and Gate 2 mixtures are the same 44 the
  after-the-fact check lists: 30 DS-1000, 6 HumanEval+, 5 LiveCodeBench, 2 BFCL, 1 MMLU-Pro. Its
  one "strong" item is the gate's one rejecting item. No row touches GPQA, IFEval or BIRD.
- **The 2 rejected rows** are two copies of one opencoder exercise: "Write a function to calculate
  the area of a triangle given the lengths of its three sides using Heron's formula". The item is
  BFCL `multiple_1`: "Calculate the area of a triangle, given the lengths of its three sides: 3, 4,
  and 5." It has 17 tokens, so 5 13-grams, and the one they share is 20%, exactly the line. That is
  not a leak of the item's task, but it is 13 of its 17 words in order. Losing two rows costs
  nothing.
- **The reasoning pilot:** one oasst1 row touches 13 DS-1000 items, each at 1.5-8% of its
  13-grams. The shared text is the sequence 1 2 3 ... 23. The request asks for a message under 140
  characters and the base's reply counts, while those DS-1000 items print runs of consecutive
  numbers. It's the kind of overlap the audit is there to show, and far below the line.
- **The new rules found nothing:** no short item appears whole in any of the three mixtures, and
  no row carries a BFCL schema.

## Where the items live

The gate needs the battery's `items/`, which the battery run keeps on dashi
(`~/benchlab/runs/2026-09-24-gate2-battery/items/`). A local copy sits in
`projects/dsbench/data/battery/items/`, and git ignores that directory: GPQA's terms forbid
republishing its examples. The reports carry item ids, never item text.

```bash
uv run python -m dsbench.sftgen.decontaminate MIXTURE.jsonl \
    --battery-items data/battery/items --report REPORT.json
```

`assemble.py --battery-items` applies the same gate while it samples. The Gate 1 and Gate 2
mixtures were assembled without it, so leaving the flag out reproduces them.

## Not covered

- **The gate protects the battery's own measurements.** It checks the items the battery runs, which
  for MMLU-Pro and LiveCodeBench are samples, not the whole benchmarks.
- **Spider dev/test are not battery items.** ADR-001 requires SQL data to be decontaminated
  against them. If the SQL pool adds Spider train (Revision 2, decision 3), they have to join the
  index.
- **Fourteen BFCL requests of 3-5 tokens stay unchecked,** as does any item text a row reproduces
  with its words changed. A 13-gram only catches copies.
