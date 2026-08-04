# NOTAM datasets — assessment & plan (Phase 7 groundwork)

**Assessed live 2026-08-04** (repos cloned + inspected, not recalled). NOTAMs are the frontier the
[sourcing study](aviation-data-sources.md) flagged: parsers fail the free-text **E-field**, so there
is no parser-consensus answer key like METAR/TAF had. The enabler is that **expert-annotated gold
already exists** — two academic datasets (same lab, Beihang/ADCC). This doc records what's actually in
them and how the NOTAM pipeline must differ.

> **Headline:** use **Knots** as the primary dataset (English labels, `raw_text → manual_fields`
> structured gold, ~10 categories, ~12K records). **NOTAM-Evolve** is a ready instruction-SFT set but
> its labels are **~88% Chinese** — a poor fit for an English decode study; keep it as a secondary /
> prompt-format reference only. The task inverts vs METAR/TAF: **the gold is given, not built** — no
> ingest/consensus/gold layer; the work is schema + harness adaptation + SFT.

## The two datasets

### Knots — PRIMARY (github.com/Estrellajer/Knots)
- **Format:** per-category JSON, `{metadata, records}`; each record = `{id, category, raw_text, manual_fields}`.
  - `id` — the NOTAM id (e.g. `MMFR_A1701/24`).
  - `raw_text` — the operational NOTAM content, A) + E) fields (e.g. `A) MMRX E)RWY 13/31 CLSD`). This
    is the model input. (Not the full ICAO NOTAM with Q-line/B/C/D — the E-field is the hard part.)
  - `manual_fields` — the expert extraction, `{rows: [ {...}, ... ]}`, category-specific (a NOTAM can
    affect multiple runways → multiple rows). Runway row: `affect_region, airport, flight_type,
    runway, status_type, ppr, aip, tora/toda/asda/lda, distance_chg`.
- **Labels are ENGLISH** (`flight_type: "international, domestic, regional"`) — the key advantage.
- **Categories (~10) + counts:** area 5,660 · light 1,450 · airport 1,015 · runway 538 · navigation
  442 · stand 432 · taxiway 369 · airway 325 · procedure 195 · (+ small test/variant files). ~12K total.
- **No explicit train/test split** — we'll make a deterministic one (by id hash), as we did for METAR/TAF.

### NOTAM-Evolve — SECONDARY (github.com/Estrellajer/NOTAM-Evolve)
- **Format:** ready instruction-SFT, `{instruction, input, output}`; 4 categories (area/light/runway/
  taxiway), train+test (~6.5K train / ~3.5K test). `instruction` is a detailed per-category extraction
  prompt (incl. a keyword rubric + chain-of-thought guidance); `input` is the raw NOTAM.
- **Labels are ~88% Chinese** (5,695 of 6,495 train outputs contain Chinese, e.g. `flight_type:
  "国际,国内,地区"`). For an English study this is a liability. Value here: the **instruction/rubric
  text** (useful for prompt design) and the pure-ASCII taxiway subset (800). Not the primary SFT source.

### PyNotam — AVOID as a dependency (github.com/slavak/PyNotam)
- **GPL-v2** (copyleft) — importing it would impose GPL on our MIT/Apache harness. Also **not needed**:
  Knots already provides the extracted fields as gold, and the Q-line (its specialty) isn't the hard
  part. Use only as a dev-time reference; if a Q-line parser is ever needed, reimplement it (structured,
  simple) or use a permissively-licensed alternative (dbrgn/notam-parse).

## ⚠️ Licensing caveat (must resolve before any publish)
Both Knots and NOTAM-Evolve READMEs carry an **Apache-2.0 badge**, but the **LICENSE file is absent**
from both repos (the badge links 404). Author *intent* is clearly Apache-2.0 (attribution), which is
fine for a private study — but before publishing any derived dataset/model we must confirm the license
(open an issue / check the papers: Knots arXiv 2511.12630, NOTAM-Evolve arXiv 2511.07982 AAAI-2026).
The datasets are also **2024-only, Asia-skewed, ADCC-annotated** — note the distribution.

## How the NOTAM pipeline differs from METAR/TAF
| layer | METAR/TAF | NOTAM |
|---|---|---|
| gold | built (parser consensus + NOAA/AWC) | **given** (Knots `manual_fields`) — no consensus layer |
| ingest | IEM/AWC backfill + collector | **git-clone Knots**, adapt `raw_text`/`manual_fields` |
| task | decode the whole report | **extract category-specific structured rows** from the E-field |
| what "beats a parser" means | fidelity to a parser | **matching expert humans** on free text parsers can't do |

## Plan (Phase 7)
1. **Schema** (`schema/notam.py`): a canonical per-category extraction record + row model. Start with
   the highest-value, best-structured categories — **runway, taxiway, area** (+ light) — each a small
   field set (airport, subject id, status, affect_region, flight_type, …).
2. **Adapt** (`ingest`/`finetune`): map Knots `{raw_text, manual_fields}` → eval records + SFT pairs.
   Deterministic by-id train/test split (disjoint), geo/category-stratified.
3. **Freeze `eval/notam/v1`** — held-out NOTAMs, sha256, manifest (same discipline).
4. **Scorer** (`score_notam`): field-level 5-way (HIT/WRONG/ABSTAIN/HALLUCINATE/TRUE_ABSTAIN) with
   **row alignment** (a NOTAM → multiple rows, like TAF periods). EM = all rows/fields right.
5. **Prompt + harness runner** (category-aware; the NOTAM-Evolve rubric text can seed the prompt).
6. **SFT (rank-16) → train on dashi → eval**; then fold into the combined adapter (toward all-four).

**Tailwind:** today's combined METAR+TAF result showed multi-task training is free/synergistic here, so
adding NOTAM to one adapter (the all-four goal) is well-motivated.

---

## UPDATE 2026-08-05 — new-sources sweep → OpenNOTAM switch + DEEL-AI

A live research sweep (HF/Kaggle/academic/GitHub + raw feeds) found two real wins; corpus grew
**4,718 → 11,340 clean English records** (0 Chinese), and a new classification task was added.

- **OpenNOTAM** (github.com/Estrellajer/OpenNOTAM) now the extraction source, superseding raw Knots.
  It is the Knots EXPERT gold reformatted as instruction/input/output SFT triples — **verified
  label-identical** (runway 538/538 match), so no quality loss. Gains: (a) `area_type` is a 6-value
  Chinese ENUM, remapped to English (AREA_TYPE_REMAP) → **area recovered (5,514 vs 797)** with its
  type; (b) **+rvr, +standard** categories (11 total); (c) a ready **train/test split** we reuse.
  `ingest/opennotam.py` replaces `ingest/knots.py`. License caveat unchanged (no LICENSE file, MIT
  badge — same lab as Knots; confirm before publishing).
- **DEEL-AI/NOTAM** (HF, **MIT**, independent French lab) — **8,478** records, 0 Chinese, a NEW task:
  single-label **classification** into 13 classes (Airspaces, GPS, Obstacles, Runway, Taxiway, …).
  Being added as a separate NOTAM classification capability + eval.
- **FAA NOTAM API** (US-Gov, ~3.73M/yr, redistribute-safe with an attribution caveat) + a permissive
  Q-line parser (`dbrgn/notam-parse`, MIT — replaces GPL PyNotam) = the durable path to scale raw
  later. Two large raw sets (jcoupon ~98K, krooonal ~96K) are clean English but **unlicensed** →
  internal bootstrap only, never redistribute.
- **Rejected:** NOTAM-Evolve (Chinese labels), AirsideLabs (QA task, no license), Kaggle (nothing
  usable), FAA DINS (403 bot-block), EUROCONTROL EAD (gated), ICAO API (paid).
