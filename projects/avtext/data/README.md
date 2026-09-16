# data/ — zones and rules

Bulk data **never** lives in git. Only two things in this tree are committed:
this README and `manifest.json`. Everything else is rebuilt by `make backfill`
or pulled from the Hugging Face dataset repo. (`.gitignore` enforces this.)

| Zone | Contents | Mutable? | Committed? |
|---|---|---|---|
| `raw/` | Exact source bytes, gzipped, partitioned `{source}/{product}/dt=.../` | **No** — immutable once written | No |
| `reference/` | Trust-anchor tables (FAA vocab, OurAirports, WMO CCT) as CSV | Rebuildable | No (tiny; could be committed later) |
| `processed/` | Normalized Parquet corpus (DuckDB reads this) | Rebuildable | No |
| `gold/` | Hand-transcribed worked examples from AC 00-45H / FMH-1 | Append-only | Seeds may be committed later |
| `third_party/` | **EVAL-ONLY quarantine** (Knots, ERAU) | — | **NEVER** (eval-only) |

`manifest.json` is the provenance ledger: for every fetched artifact it records
the URL, sha256, byte size, and fetch time. It is the one committed record that
makes the corpus reproducible and auditable.
