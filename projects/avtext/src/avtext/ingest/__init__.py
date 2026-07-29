"""ingest — automated data acquisition (Skill A, Phase 1).

Fetch raw aviation text and record provenance. The invariant: raw bytes are
*immutable* once written, every artifact has a sha256 in data/manifest.json, and
`make backfill` can rebuild data/ from nothing.

Planned modules:
    iem.py         one-off history backfill (Iowa Environmental Mesonet ASOS/AFOS)
    awc.py         6-hourly snapshots of AWC caches (raw + official decode)
    references.py  trust-anchor pack: FAA vocab, OurAirports, WMO CCT, gold PDFs
    zenodo.py      NOTAM bulk (deferred until the METAR/TAF slice is done)

Pattern per source: fetch -> verify + record sha256 -> write immutable
data/raw/{source}/{product}/dt=.../*.gz -> normalize to Parquet in data/processed/.
"""
