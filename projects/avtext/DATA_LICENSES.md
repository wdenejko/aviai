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
| `Estrellajer/OpenNOTAM` (GitHub) | NOTAM extraction train + eval (11,340 clean) | No LICENSE file upstream (re-verified 2026-09-11, GitHub license API 404). **Cleared by the project owner on 2026-09-11: use and model publication permitted under a permission held by the project** — record the grant reference (issue/email/text) here and in the model card. | ⚠️ | 9,049 records in the S24 training mix (`train_notam.jsonl`, `train_all_lg.jsonl`). Redistribute models trained on it, not the records themselves, unless the grant says otherwise. |
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
| NOTAM extraction `Estrellajer/OpenNOTAM` (= Knots) | No public license; **owner-held permission (2026-09-11)** | ✅ (with the grant cited) | Was the blocker for the Phase-7 adapter; cleared for the S24 grown adapter. |

**Consequence (updated 2026-09-11).** With the OpenNOTAM permission in hand, the S24 grown all-products
adapter (`adapter-all-lg`, 49,214 examples) is publishable as is: Apache-2.0 for the adapter, the Gemma
PUP linked, attribution to Google DeepMind, IEM/AWC, DEEL-AI (MIT) and OpenNOTAM (permission). Release
package: `release/gemma-4-e4b-avtext-lora-r16/` (weights pulled from dashi, never committed). The
licensed-only mix (`build_sft_licensed.py`) remains the fallback if the permission is ever withdrawn.
