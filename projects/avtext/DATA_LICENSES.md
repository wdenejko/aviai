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
| `Estrellajer/Knots` (GitHub) | held-out eval | **unclear** | ⛔ | Quarantine in `data/third_party/`. **TODO: open a license issue upstream.** |
| ERAU GA-Wx display interpretation | held-out eval | academic collection | ⛔ | Eval-only; never redistributed. |

## The quarantine rule (`data/third_party/`)

Anything in `data/third_party/` is **eval-only** and **must never** be:
committed to git, uploaded to the HF dataset repo, included in any training
mix, or copied into `data/processed/`. It exists solely so the harness can score
against it locally. `.gitignore` already excludes the directory — keep it that
way.
