# dsbench v2 — agentic aviation sandbox

The live data stack a tool-calling agent operates on, for real-world-facing aviation
DE/DA/DS benchmarking. Design: [`docs/adr/ADR-003-agentic-aviation-sandbox.md`](../docs/adr/ADR-003-agentic-aviation-sandbox.md).

**Phase 1 (now):** ClickHouse warehouse + a Python workspace. **Phase 2:** live Airflow + MLflow.

## Services

| Service | Role | Ports (localhost) |
|---|---|---|
| `clickhouse` | warehouse; holds the aviation datasets; the agent's `run_sql` target | 8123 (HTTP), 9000 (native) |
| `workspace` | Python container; the agent's `run_python` executes **here**, not on the host | — |

Creds (local only): user `avbench`, password `avbench`, db `aviation`.

## Use

```bash
cd projects/dsbench/sandbox
docker compose up -d clickhouse                     # warehouse
docker compose exec clickhouse clickhouse-client -u avbench --password avbench -q "SELECT version()"
docker compose up -d --build workspace              # + agent workspace (for run_python / ingest)
docker compose down                                 # stop (add -v to also wipe the ch-data volume)
```

HTTP query from the host:
```bash
curl -s 'http://localhost:8123/?user=avbench&password=avbench' --data-binary "SELECT count() FROM aviation.flights"
```

## Data

Loaders live in `ingest/`; snapshots are pinned in `manifests/`. Loaded tables in `aviation`:

| table | rows | source | license | loader |
|---|---|---|---|---|
| `flights` | 607,577 | BTS On-Time Performance, June 2026 (PREZIP) | US-gov public domain | `bts.py --year 2026 --month 6` |
| `metar` | 45,796 | IEM ASOS archive, June 2026, 5 hubs | US NWS public domain | `metar.py --year 2026 --month 6` |
| `taf` | 9,060 | IEM TAF archive, June 2026, 5 hubs (decoded per period) | US NWS public domain | `taf.py --year 2026 --month 6` |
| `notam` | 8,478 | DEEL-AI/NOTAM (13-class) | MIT (publishable) | `notam.py` |
| `airports` | 5 | hub IATA↔ICAO dimension | — | (created by `metar.py`) |

Run a loader with `uv run --package dsbench python ingest/<loader>`. Flights + METAR + TAF align on
June 2026 and the hub set (ATL/ORD/DFW/DEN/LAX) so cross-source problems work. **NOTAM is a static
~2024 corpus, not date-aligned and with no airport/time key** — a standalone reference corpus for
NOTAM text/classification tasks (ADR-003 §4). Bulk raw stays in `.data/` (gitignored); only manifests
are committed.

## Safety / trust model

The agent executes **arbitrary Python and SQL**. It is contained to the `workspace` and
`clickhouse` containers on the `avnet` network — never the host. This is for **your own model on
your own box**; do not point the harness at an untrusted endpoint. The only host mount is the repo
working directory (agent files, `dags/`, ingest scripts).
