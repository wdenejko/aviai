# Anatomy of the v2 "7% hallucination" — what the finetuned E4B actually fabricates

**Analysed: 2026-08-01.** Source: the frozen rank-16 v2 run
`reports/gemma-4/runs/20260731T170322Z-gemma-4-e4b-it-q8-v2-r16/` (E4B + rank-16 LoRA, 6,200
records, eval/v2 sha `6e19b0eb…`). Method scripts: `scratchpad/halluc_drill.py`,
`scratchpad/halluc_classify.py` (kept out of the repo; the numbers below reproduce from the run's
`scores.jsonl` joined to `eval/v2/eval.jsonl` by record id).

> **Headline:** the widely-quoted "hallucination stays ~7% even after finetuning" is **mostly an
> eval-labelling artifact, not model behaviour.** At least **70% of the flagged fabrications are
> cases where the model read a value correctly and all three parsers failed** — so the gold is empty
> and the scorer calls the model's *win* a hallucination. The real, dangerous fabrication rate is
> **≤ 2.1%**, and it collapses to a handful of recognisable, fixable modes.

## 1. Where the number comes from

`hallucination_rate = HALLUCINATE / (HALLUCINATE + TRUE_ABSTAIN)` — the fraction of *truly-absent*
fields on which the model nonetheless emitted a value. For the r16 run:

```
HALLUCINATE = 625 fields   across 285 records
TRUE_ABSTAIN = 8027
rate = 625 / 8652 = 7.2%
```

A field is HALLUCINATE when `reference[field] is None` (gold says absent) but `prediction[field]` has
a value (`score.py:52`). The load-bearing assumption is **"gold None ⇒ the field is genuinely
absent."** That assumption holds on `clean`/`dissent` records (the parsers succeeded, so an empty
field really is empty). It **breaks on `parse_fail` records** — there `reference` is None *because the
parser consensus itself failed*, which is not the same as "the value isn't in the raw."

That distinction is the whole story: **616 of the 625 hallucinations (98.6%) are on the `parse_fail`
split.**

| field | count | share | split concentration |
|---|--:|--:|---|
| altimeter_hpa | 189 | 30.2% | 189 parse_fail |
| visibility_m | 78 | 12.5% | 75 parse_fail, 3 clean |
| temperature_c | 76 | 12.2% | 76 parse_fail |
| dewpoint_c | 67 | 10.7% | 67 parse_fail |
| clouds | 54 | 8.6% | 54 parse_fail |
| wind_dir | 40 | 6.4% | 35 parse_fail, 5 clean |
| wind_speed | 38 | 6.1% | 38 parse_fail |
| report_type | 27 | 4.3% | 27 parse_fail |
| automated | 27 | 4.3% | 27 parse_fail |
| cavok | 27 | 4.3% | 27 parse_fail |
| wind_gust | 2 | 0.3% | 1 parse_fail, 1 clean |

## 2. Reclassifying the 625 — recovered win vs. real fabrication

For every flagged field we ask a single, conservative question: **is the fabricated value actually
present in the raw METAR?** If yes, the model *recovered* a value the parsers dropped (a mislabelled
win); if we cannot positively locate it, we count it as *fabricated* (a real hallucination). The test
is deliberately biased against the model — a recovery is only credited when the value is provably
there (`Q1016` in the raw for `altimeter=1016`, the `TT/DD` group for temp/dew, a literal `METAR`
token for `report_type`, etc.). Anything ambiguous is scored as fabrication.

```
RECOVERED  (model read it, parsers failed) : 439  (70.2%)   <- mislabelled WIN
FABRICATED (masked/garbage, invented)      : 186  (29.8%)   <- real hallucination
```

Because the classifier is conservative, these are bounds, not point estimates:

- **≥ 70% recovered** is a **floor** — several genuine recoveries are miscounted as fabrication
  (e.g. `26//` → temp 26 correct but dew masked; `050//KT` → dir 050 correct but speed masked), so
  the true win share is higher.
- **≤ 2.1% fabrication** is a **ceiling**: `186 / 8652 = 2.15%`, and the truly-invented-from-nothing
  subset is smaller still (some of the 186 are *misreads of a present-but-corrupted value*, below).

**The corrected picture: value accuracy 99.4% stands; the real hallucination rate is ~2%, not 7%.**

### 2a. The recovered wins — the LLM-beats-parser evidence, hiding in the "bad" column

These are the reports where the three parsers (python-metar, avwx, mivek) all failed, so gold is
empty, but the model read the data correctly. Two dominant corruption patterns defeat the parsers and
not the model:

- **Glued tokens (missing spaces):** `SGCO … 15/14Q1016` → model: temp 15, dew 14, altimeter 1016 —
  all correct; parsers choke on `14Q1016`. Also `13/06Q1024`, `22///Q1013`, `26// Q1014`.
- **Doubled / prefixed station ids:** `MGHT METAR MGPB 250600Z …`, `COR METAR SEGU 180400Z …` —
  the header re-parse fails, so the whole record drops to parse_fail; the model reads every field.

This is exactly the thesis the messy-tail eval was meant to test, showing up unbidden: **on
deformed real-world input the finetuned model extracts data the deterministic parsers cannot.** The
eval simply has no way to score it as a win, because on `parse_fail` records `gold = None` and
`model = correct value` is indistinguishable from fabrication.

### 2b. The real fabrications — four recognisable, fixable modes

The ≤186 genuine fabrications are not random noise. They cluster:

1. **Masked-sensor fill-in (the dominant, most dangerous mode).** A field is explicitly masked with
   slashes — the station is *reporting that the sensor is unavailable* — and the model overwrites that
   explicit "unknown" with a confident, plausible value:
   - `EHDL … AUTO /////KT //// // ////// 14/12 Q1021` → visibility fabricated 10000 (raw vis is
     `////`).
   - `KJFK … AUTO ///18G24KT 10SM CLR …` → wind_dir fabricated **180** (direction masked `///`; the
     model appears to promote the *speed* digits `18` to a direction).
   - `KJFK … ///11KT …` → wind_dir fabricated 11 (again reading the masked-dir slot from the speed).

   This is the one mode that also shows up on the trustworthy `clean` split, which makes it the
   highest-confidence real defect. It is also the most learnable: `///`/`////` has a single, crisp
   meaning ("not available") that the target should map to `null`.

2. **QFE→QNH confusion.** Station MGCB (a high-elevation field) reports `QFE 872.0` (pressure *at
   field elevation*, not the sea-level altimeter). The schema correctly stores `altimeter_hpa = None`
   (QFE is not the altimeter). The model, having internalised that altimeters live near ~1013 hPa,
   fabricates a plausible sea-level value (`1008`, `1011`, `1018`). ~10 cases, one station family —
   a genuine, meaningful hallucination: it invents a number it has no basis for.

3. **Split-token misread.** `OPIS … Q10 06 …`, `OPPS … Q100 2`, `OPLA … Q10 04` — the altimeter is
   present but broken across a space. The model attempts it and gets it *wrong* (reads `Q10 06` as
   **1000**, not 1006). Strictly this is a WRONG value, not invention-from-nothing, but with no gold
   it scores as HALLUCINATE. ~15–20 cases.

4. **Garbage confabulation (rare, scariest).** A transmission is corrupted into noise —
   `EQYS 211946Z AUTO 25z 5j RJ*S218 T02500172 TSNO $` — and the model emits a *full* plausible
   record (wind 25@10, vis 1000, temp −2/dew −17, altimeter 1017). A handful of records, many fields
   each. This is fabrication in its purest form and the clearest argument for an "if it doesn't look
   like a METAR, abstain on everything" behaviour.

## 3. Two consequences

### 3a. Methodological — the HALLUCINATE metric is untrustworthy on `parse_fail`

On the `parse_fail` split, `gold = None` conflates *"the value is truly absent"* with *"the parser
consensus couldn't read it."* The five-way scorer therefore **cannot distinguish the model's biggest
win (reading through corruption) from its worst failure (fabricating from nothing)** — both look like
`gold None, pred value`. Every headline hallucination number that includes `parse_fail` is inflated
by the recovery wins.

This is precisely the gap the **messy-tail / human-gold eval** was queued to close: to score the
`parse_fail` tail honestly we need a gold source that is *not* the parser consensus — i.e. a
human/expert (or a stronger frontier-model) decode of the deformed reports, so "absent" and
"parser-couldn't" become separable labels. Topic 2 has, in effect, produced the strongest motivation
yet for that eval — and a ready-made ~285-record candidate pool (every record with a flagged field)
to label first.

### 3b. Practical — the fix is targeted abstention, NOT "abstain more on messy input"

The naive reaction ("hallucination is 7%, make the model abstain more on `parse_fail`") would be
**actively harmful**: it would suppress the 70% recovery wins to remove the 2% fabrication. The
correct, narrow intervention follows the taxonomy:

- **Masked-field augmentation (addresses mode 1, the bulk).** Take clean training METARs, randomly
  replace a field's tokens with the slash-mask the standard uses (`///`, `////`, `//`, `/////KT`),
  and set that field's target to `null`. This teaches the crisp rule *slash-mask ⇒ absent ⇒ null*
  without touching the model's ability to read well-formed fields. High precision, cheap to generate.
- **QFE handling (mode 2).** Add training examples with `QFE` pressure groups whose `altimeter_hpa`
  target is `null` — teach that QFE is not the altimeter.
- **Garbage-guard (mode 4).** A small number of noise-in → all-null-out examples to establish
  "doesn't parse as a METAR ⇒ abstain," rather than confabulate a record.
- **Leave the glued-token / doubled-id recoveries alone** — those are wins; do not train them toward
  abstention.

Expected effect: the honest (human-gold) hallucination rate drops toward the mode-1/2/4 residue while
the recovery wins are preserved — and, unlike the current metric, we would be able to *measure* it.

## 4. One-line summary for the log

> The v2 "7% hallucination" is ≥70% mislabelled parser-defeat wins and ≤2.1% real fabrication; the
> real fabrication is dominated by **slash-masked fields the model fills in instead of leaving null**,
> which is fixable with targeted masking augmentation — and the finding is itself the strongest case
> for building the human-gold messy-tail eval, since the current scorer literally cannot tell the
> model's best behaviour from its worst on the `parse_fail` split.
