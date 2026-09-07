# Data licenses & redistribution rules

**This file is part of code review.** Before any source is added to `ingest/` or
any artifact is committed/uploaded, its row must exist here with a
redistribution decision. Guardrail from the plan (§5.4): *Knots and ERAU never
leave `data/third_party/`; license leaks fail review.*

Legend for **Redistribute?**: ✅ yes · ⚠️ derived/with-attribution only · ⛔ never
(eval-only quarantine).

## Corpus & reference sources

| Source | Used for | License / terms | Redistribute? | Notes |
|---|---|---|---|---|
| IEM ASOS/AFOS archive | METAR/TAF/SIGMET raw history | Underlying obs are US NWS public-domain; IEM asks for polite use + attribution | ⚠️ | We commit only checksummed *manifests*, never bulk. Bulk → HF dataset repo. |
| AWC caches + API (aviationweather.gov) | current raw + officially decoded snapshots | US Gov public domain; ToU requires a custom `User-Agent` and polite request rates | ⚠️ | Collector must set a descriptive UA. ~15-day history window = why we collect early. |
| FAA Q-codes (Appendix B), Contractions 7340.2, PCG, NASR | trust-anchor vocab/registries | US Gov — public domain | ✅ | Tiny; safe to commit as CSV under `data/reference/`. |
| OurAirports CSVs | station registry | Public domain (explicit) | ✅ | |
| WMO `wmo-im/CCT` | code tables (trust anchor) | MIT | ✅ | Clone/vendor is fine. |
| WMO `wmo-im/iwxxm` schemas | XML schema reference | _verify before use_ | ⚠️ | Confirm license text in Phase 2. |
| Zenodo NOTAM (Pik 2023, `11420433`) | NOTAM bulk (later phase) | CC-BY 4.0 | ⚠️ | Attribution required; defer until slice is done. |
| AC 00-45H, FMH-1 (2019), NWS METAR decode key | **gold-seed** worked examples | US Gov — public domain | ✅ | We transcribe examples (copy-only, zero judgment). |

## Eval sets

| Source | Used for | License / terms | Redistribute? | Notes |
|---|---|---|---|---|
| HF `AirsideLabs/pre-flight-06` | held-out eval | MIT | ✅ | |
| `Estrellajer/Knots` (GitHub) | NOTAM extraction gold | **NONE** — no LICENSE file (verified via GitHub API 2026-09-07; the README's Apache-2.0 badge 404s) | ⛔ | All rights reserved by default. Label-identical to OpenNOTAM below. Open a license request upstream. |
| `Estrellajer/OpenNOTAM` (GitHub) | NOTAM extraction train + eval (11,340 clean) | **NONE** — no LICENSE file (verified 2026-09-07) | ⛔ | ⚠️ **9,083 records were used in Phase 7 TRAINING** (`train_notam.jsonl`, `train_all.jsonl`) → the all-products adapter is **not publishable**. See *Model release audit*. |
| ERAU GA-Wx display interpretation | held-out eval | academic collection | ⛔ | Eval-only; never redistributed. |

## The quarantine rule (`data/third_party/`)

Anything in `data/third_party/` is **eval-only** and **must never** be:
committed to git, uploaded to the HF dataset repo, included in any training
mix, or copied into `data/processed/`. It exists solely so the harness can score
against it locally. `.gitignore` already excludes the directory — keep it that
way.

## Model release audit (2026-09-07)

Verified before any Hugging Face publication of an avtext adapter. Sources checked live
(GitHub API license field, HF model/dataset cards, ai.google.dev).

| Component | License | Publishable? | Notes |
|---|---|---|---|
| Base `google/gemma-4-E4B-it` (via `unsloth/gemma-4-E4B-it`) | **Apache-2.0** + Gemma Prohibited Use Policy / Intended Use Statement | ✅ | Gemma 4 has its **own** license — the Gemma 1–3 "Terms of Use" explicitly exclude it. Ship the Apache-2.0 notice + attribution to Google DeepMind and link the PUP. The PUP has no aviation/safety-critical clause (only "automated decisions affecting rights/welfare"); a research-use disclaimer covers it. |
| METAR / TAF gold (IEM ASOS, AWC) | US Gov public domain | ✅ | Attribute IEM + AWC. |
| NOTAM classification `DEEL-AI/NOTAM` | **MIT** (verified) | ✅ | Attribute DEEL-AI. |
| NOTAM extraction `Estrellajer/OpenNOTAM` (= Knots) | **NONE** | ⛔ | **Blocker.** A model trained on it cannot carry a clean license. |

**Consequence.** The Phase-7 all-products adapter includes OpenNOTAM and is **study-only**. The
publishable artifact is an adapter retrained on the **licensed-only mix** — METAR + TAF + DEEL-AI
classification (`finetune/build_sft_licensed.py` → `train_licensed.jsonl`). NOTAM *extraction* is
dropped from the published model unless/until upstream grants a license (request opened separately).
