# Reasoning pilot: the base model's own thinking-mode traces, measured

**Date:** 2026-09-28 · raw numbers: `20260928-reasoning-pilot.json` · code:
`src/dsbench/sftgen/reasoning_pilot.py` · raw generations, logs and the box scripts:
`~/benchlab/runs/2026-09-28-qwen36-reasoning-pilot/` on dashi

**Status: done. Three answers for the retrain (ADR-001 Gate 2 item 5).**
- Train at **8192 tokens**: 95% of the traces fit whole, against 85% at 4096.
- A thinking-on row costs about **six times the tokens** of a Gate-2 row, so the mixture shrinks
  in rows.
- **Target A cannot come from the base's own traces.** It gets the targeted ClickHouse
  conventions wrong 72% of the time, systematically, and one of Target A's four families checks
  answers against a wrong truth.

## What was run

The bare base (the I-Mini GGUF the adapter trains on, no LoRA) answered 200 single-turn prompts from
the Gate-2 mixture on the production llama-server build:
- thinking on per request;
- Qwen's thinking-mode sampling (temperature 0.6, top-p 0.95, top-k 20);
- up to 16,384 tokens per reply;
- 8 slots.

| Pool | Prompts | What they ask |
|---|---:|---|
| Target A | 80 (ClickHouse 32; Postgres, MySQL, DuckDB 16 each) | SQL-dialect date/time conventions |
| gretel_sql | 32 | text-to-SQL with CREATE/INSERT context |
| tulu3 | 40 | general instructions (replay) |
| opencoder_edu | 32 | textbook coding exercises |
| swe_swiss | 16 | SWE sub-tasks: file localisation, test writing, patches |

The Target A prompt asks for the SQL only. The Gate-2 prompt also asked for "the numeric result",
which the model cannot know without the data, and the adapter learned to invent one. The ClickHouse
answers ran in the dsbench ClickHouse sandbox on each row's own synthetic table, rebuilt from its
seed. Before any model answer was judged, the row's gold SQL had to reproduce the stored truth on
the rebuilt table; all 32 did. The Gate-2 slice turned out to be generated with 1,500 rows per
table, not the generator's default 4,000.

## Lengths

A training row is the prompt, the reasoning, the answer and the end token:

| Pool | Reasoning, median | Reasoning, p90 | Row, median | Row, p90 | Row, max | ≤ 4096 | ≤ 8192 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Target A | 1,556 | 4,354 | 1,673 | 4,504 | 16,500 | 71/80 | 76/80 |
| gretel_sql | 1,249 | 2,238 | 1,599 | 2,769 | 4,383 | 31/32 | 32/32 |
| tulu3 | 1,384 | 3,296 | 2,074 | 4,529 | 7,634 | 35/40 | 40/40 |
| opencoder_edu | 2,219 | 9,289 | 2,807 | 10,246 | 13,522 | 25/32 | 26/32 |
| swe_swiss | 2,438 | 4,654 | 4,336 | 6,227 | 7,376 | 7/16 | 16/16 |
| **all** | | | 2,085 | | | **169/200** | **190/200** |

A reply averages 2,560 generated tokens. The Gate-2 mixture's assistant turns had a median of 94,
and Target A's "reasoning" was one sentence of about 46 tokens.

The rows over 8,192 are opencoder's verbose tail and four Target A rows. The opencoder cases are
8-12k tokens of re-checking on classic exercises (meeting rooms, Roman numerals). All of them end
with an answer, and their text is only mildly repetitive (zlib ratio 0.26-0.29 against the
pool's median 0.36). One reply hit the 16,384-token limit: a Target A MySQL timezone question
where the model kept "refining" the same query (zlib 0.20). That is 1 loop in 200.

## Generation throughput

511,908 generated tokens in 67 minutes of wall time: **127 tokens/s** across 8 slots, 16.6 per
slot. A million generated tokens costs about 2.2 hours of box time.

## Target A: the base is wrong on exactly the conventions the target teaches

| Family (ClickHouse) | Verified | Wrong | Error |
|---|---:|---:|---:|
| month-bucket | 7 | 0 | 0 |
| weekend-flag | 1 | 12 | 0 |
| weekday-numbering | 1 | 5 | 0 |
| timezone-direction | 0 | 2 | 4 |
| **all** | **9 (28%)** | 19 | 4 |

The misses are not noise; they are one belief. After 1-2k tokens of reasoning, the base concludes
that ClickHouse's `toDayOfWeek` numbers Sunday as 1, as MySQL's `DAYOFWEEK` does, where ClickHouse
is ISO: Monday = 1, Sunday = 7. So it writes `IN (1, 7)` for the weekend and `= 6` for Friday.
Rejection sampling keeps only the answers the base already gets right. More samples per prompt
would find the rare right one in these families, but the kept traces would teach the target
nothing the base does not already believe.

The timezone family fails for two reasons. The base invents functions: 4 of its 6 answers call a
nonexistent `toDateTimeInTimezone`. **The family's truth is also wrong for real data.** The
generator converts to UTC with fixed summer offsets (New York +4, Los Angeles +7) for timestamps
spread over the whole of 2025. The prompt states neither the offsets nor that daylight saving is
to be ignored. A correct, DST-aware conversion therefore disagrees with the truth for about five
months of the data, and the 841 Gate-2 rows of this family taught the fixed-offset CASE as the
answer.

## What it means for the retrain

1. **Sequence length 8192.** At 4096, 15% of rows would not fit, including 9 of 16 SWE rows.
   Dropping them would cut the long tail, which is the brevity the battery punished; at 8192,
   5% go, mostly verbose deliberation and the one loop. The step costs 1.62× the tokens/s of
   2048 (`20260928-seq4096-enablement.md`).
2. **Budget in rows, not tokens.** At about 3k tokens a row, the Gate-2 budget of 10M tokens is
   about 3,300 rows, and 15 hours of training at 182 tokens/s. Generating those rows once takes
   about 18 hours on the box, before any rejection overhead.
3. **Replay and breadth: the base's own traces.** For tulu3, opencoder and SWE, the base's
   thinking-mode answers need no verification to serve as replay. They are the model's own
   distribution, so they preserve its reasoning length by construction. Replies that hit the
   token limit are dropped.
4. **Target A needs a different source of reasoning.** Two candidates:
   - hint-conditioned generation (STaR's rationalisation): state the convention in the
     generation prompt, keep traces whose SQL verifies and which do not quote the hint, and
     train on them without it;
   - a teacher that knows the conventions.

   The timezone family has to be fixed first, either by stating the offsets in the prompt or by
   computing the truth with real time-zone rules.
5. **Verify the other dialects too.** Postgres, MySQL and DuckDB answers were measured for
   length only. They need their own sandboxed engines before any of them become training rows.
