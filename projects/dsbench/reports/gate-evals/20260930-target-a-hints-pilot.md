# Target A hint pilot: telling the base the convention

ADR-004 Revision 2, action item 2. Run 2026-09-30 on dashi in one GPU window. The code is
`sftgen/target_a_hints.py`, the raw outputs are in `~/benchlab/runs/2026-09-30-qwen36-target-a-hints-pilot/`,
and the numbers are in `20260930-target-a-hints-pilot.json`.

## Summary

- **The gap is ClickHouse's weekday numbering, and nothing else Target A tests.** Without a hint,
  the base verified:
  - 1 of 12 ClickHouse weekday rows, and 0 of 8 ClickHouse weekend rows whose question is
    unambiguous;
  - every one of its 24 ClickHouse weekday and weekend traces considers Sunday = 1.

  Month buckets verified 48 of 48, and the timezone family, now that it states its offsets, 44 of
  48. Postgres and MySQL answered their weekdays right.
- **Stated in the prompt, the convention fixes the answers:** 48 of 48 weekday rows and 32 of 32
  unambiguous weekend rows verify.
- **But the base reads it back.** A thinking trace narrates what the prompt says, and it cites
  the hint.
  - Stated plainly (pass 1), 11 of 96 weekday and weekend traces are clean.
  - Framed as the base's own knowledge (pass 2), only 4 of 96 are clean, because the base quotes
    the framing.
  - The clean traces are short: 304-940 reasoning tokens in pass 1, where the base's own weekday
    traces run to a median of 1,330.

  Hint-conditioned generation as designed can't supply Target A's rows.
- **A question wording read as GROUP BY.** "How many orders fall on a weekend (Sat or Sun), by
  order_ts?" was answered with `GROUP BY order_ts` 30 times in 32, plain or hinted. That, not a
  convention, was most of the weekend misses outside ClickHouse. The wording is fixed in
  `conventions.py`.
- **The citation filter was calibrated on these traces:**
  - it catches the attributions they contain;
  - it flags none of 272 traces that never saw a hint;
  - "the prompt says" alone can't be the test, since the base writes it all the time about the
    question.

## What was run

- **Rows:** 192 Target A rows from `dialect_conventions.py`: 4 dialects, 4 families, 12 each,
  seed 20260930, tables of 1,500 rows. Every row's gold SQL verified on its engine.
- **Serving:** the base (APEX I-Mini, no LoRA), thinking on, with Qwen's sampling
  (temperature 0.6, top-p 0.95, top-k 20), 8 slots, at most 16,384 tokens a reply.
- **Pass 1** (103 minutes, 801,681 tokens, 129.5 tokens/s): each row plain, and again with the
  convention stated in the system prompt, before its answer line. That is 384 replies.
- **Pass 2** (21 minutes, 165,616 tokens): the 96 weekday and weekend rows again. The hint was in
  its own paragraph, after "Treat the following as your own knowledge of the engine. Don't quote
  it or mention where it came from:".
- **Checks:** every reply was checked by `target_a_hints verify` on the sandboxed engine. A reply
  counts when it finished with reasoning, is one sql block and nothing else, and returns the
  row's truth. A hinted reply is kept only if it also doesn't read as told.

## The base alone

Verified replies of 12, plain:

| Family | ClickHouse | DuckDB | Postgres | MySQL |
|---|---:|---:|---:|---:|
| weekday-numbering | 1 | 9 | 11 | 12 |
| weekend-flag | 0 | 6 | 7 | 8 |
| month-bucket | 12 | 12 | 12 | 12 |
| timezone-direction | 11 | 9 | 12 | 12 |

**ClickHouse.**
- Ten of the eleven weekday misses filter on the wrong number; the eleventh grouped.
- Of the 12 weekend misses, 10 write `IN (1, 7)` and 2 invent `isWeekend`.
- The one ClickHouse success reasoned right: `dayOfWeek` returns 1 for Monday, checked on a
  Sunday it knew (29 October 2023). The base holds the right belief, just rarely.

**The weekend column is mostly a question wording.** One of the family's three phrasings ends
", by order_ts?", and 15 of the 16 plain answers to it grouped by the timestamp.
- Postgres and MySQL made no convention miss: all their misses are that phrasing, or a reply that
  ran out of tokens.
- DuckDB's convention misses are few: 2 wrong weekday numbers, 1 wrong weekend number, and an
  invented `is_weekend`.

**Timezone.** With the offsets stated (ADR-004 Revision 2), the family verifies 44 of 48. The
pilot of 2026-09-28 scored it 0 of 17, but those questions stated no offsets.

## With the hint (pass 1)

Verified replies of 12, hinted:

| Family | ClickHouse | DuckDB | Postgres | MySQL |
|---|---:|---:|---:|---:|
| weekday-numbering | 12 | 12 | 12 | 12 |
| weekend-flag | 8 | 8 | 8 | 9 |
| month-bucket | 12 | 12 | 12 | 12 |
| timezone-direction | 10 | 12 | 11 | 12 |

- Every weekend miss is the ", by order_ts?" phrasing, grouped. On the other two phrasings, 32 of
  32 verify.
- 2 of the 24 ClickHouse traces still consider Sunday = 1, and then follow the prompt.
- Reasoning gets shorter, by the median number of tokens:

  | Family | Plain | Hinted |
  |---|---:|---:|
  | weekday | 1,330 | 817 |
  | weekend | 1,776 | 1,283 |
  | month | 973 | 707 |
  | timezone | 3,678 | 3,370 |

**What the traces do with the hint.** Of 192 hinted traces, 15 are clean (weekday 8, weekend 3,
month 3, timezone 1). The rest cite it in three ways:
- they copy it: "the prompt explicitly states: "In ClickHouse, toDayOfWeek(date) numbers Monday 1
  through Sunday 7." So I must follow the prompt's definition";
- they attribute its terms to the prompt ("The prompt explicitly says `toDayOfWeek(date)` …");
- they name it ("matches the prompt's hint").

A training row has no hint in its prompt, so each of these would teach the model to cite text
that isn't there.

## Framed as the base's own knowledge (pass 2)

The same 96 weekday and weekend rows, with the framing above:
- 80 verify, much as in pass 1 (81);
- 4 are clean, against 11 in pass 1;
- the base quoted the framing itself ("Wait, the prompt says "Treat the following as your own
  knowledge of the engine…""), and 71 traces named the hint or the framing;
- the clean traces are longer (805-1,285 reasoning tokens), but too few to matter.

## The citation filter

A reply reads as told when it:
- names a hint or a note;
- talks about the framing ("own knowledge", "don't quote");
- attributes the hint's own terms (a function name, "ISO") to the prompt, as in "the prompt
  explicitly says toDayOfWeek …" or "as mentioned in the prompt";
- copies 8 tokens of the hint's wording.

**What the pilot changed.** The first version missed several of pass 1's forms: an adverb before
the verb ("explicitly says"), "the prompt's hint", "matches the prompt:", and short quotes.

**Why a phrase can't reject on its own.** The base writes about "the prompt" constantly, about the
question, in traces that never saw a hint:
- "the prompt says" appears 211 times in the 2026-09-28 pilot's 80 traces;
- across 150 such traces, "explicitly says" appears in 25 and "a hint" in 9 ("by opened_ts might
  be a hint").

So a "says" phrase counts only when the hint's terms follow it in the same clause.

**False positives:** none. That covers 272 traces that never saw a hint: 80 from the 2026-09-28
pilot and this pilot's 192 plain replies.

## What it means for the retrain

1. **Target A's hinted rows are ClickHouse's weekday and weekend rows**, plus at most a few in
   DuckDB. The other cells can come from the base's own verified plain traces: 146 of this pilot's
   192 verify, and they are the base's full length.
2. **Hint-conditioned generation doesn't work as designed.** At 4-11% kept, 250 ClickHouse rows
   would need 2,300-6,300 hinted generations. The pass-1 survivors run less than half the base's
   usual length (a median of 557 reasoning tokens against 1,330), the brevity Gate 2 punished.
   Two routes to test in the next GPU window, each about half an hour:
   - **Reasoning prefill.** Open the base's own thinking block with the convention, as a first
     sentence, and let it continue. There is no prompt text to cite, and every token after the
     first sentence is the base's. This needs a raw completion with the rendered template instead
     of the chat endpoint.
   - **Plain sampling at scale.** The base verified 1 ClickHouse weekday row of 12, and reasoned
     right there. Sampling 8-16 replies per ClickHouse prompt measures how rare the right belief
     is, and the traces it keeps are fully the base's own.

   A teacher (decision 2's alternative, Ling) stays the fallback.
3. **Fixed on the way:**
   - the ", by {col}" question wording;
   - the citation filter;
   - `sql_only` is now the answer check for every Revision 2 SQL pool.
