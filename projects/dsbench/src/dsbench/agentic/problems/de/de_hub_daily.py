"""DE (table-graded, cross-source): build a daily flights x weather fact table for the hubs.

The agent must join three sources (flights by IATA -> airports -> METAR by ICAO), bucket METAR
timestamps to days, and write a per-(airport, day) table. This is the first genuinely cross-source
agentic task. Grading is state-based: check() recomputes the expected fact INDEPENDENTLY in pandas
(a different code path from the agent's SQL and from the reference's SQL), then compares the table
the agent built. reference() builds it via SQL (the oracle gate).
"""
from __future__ import annotations

from dsbench.agentic.schema import GradeContext

HUBS = ("ATL", "ORD", "DFW", "DEN", "LAX")
_IN = "(" + ", ".join(f"'{h}'" for h in HUBS) + ")"
COLS = ["iata", "day", "n_dep", "avg_dep_delay", "n_metar"]


def _expected(ctx: GradeContext):
    """Truth, computed in pandas from raw-ish pulls — deliberately not the agent's SQL path."""
    import pandas as pd

    f = ctx.client.query_df(
        f"SELECT Origin AS iata, FlightDate AS day, DepDelayMinutes FROM aviation.flights "
        f"WHERE Origin IN {_IN}"
    )
    f["day"] = pd.to_datetime(f["day"]).dt.strftime("%Y-%m-%d")
    fd = f.groupby(["iata", "day"]).agg(
        n_dep=("DepDelayMinutes", "size"),               # all departures (incl. cancelled)
        avg_dep_delay=("DepDelayMinutes", "mean"),        # mean skips NULLs, like ClickHouse avg()
    ).reset_index()
    fd["avg_dep_delay"] = fd["avg_dep_delay"].round(2)

    m = ctx.client.query_df(
        "SELECT a.iata AS iata, m.valid_utc FROM aviation.metar m "
        "INNER JOIN aviation.airports a ON a.icao = m.station"
    )
    m["day"] = pd.to_datetime(m["valid_utc"]).dt.strftime("%Y-%m-%d")
    md = m.groupby(["iata", "day"]).size().reset_index(name="n_metar")

    exp = fd.merge(md, on=["iata", "day"], how="left")
    exp["n_metar"] = exp["n_metar"].fillna(0).astype("int64")
    exp["n_dep"] = exp["n_dep"].astype("int64")
    return exp


def check(ctx: GradeContext) -> tuple[bool, str]:
    import pandas as pd

    try:
        got = ctx.client.query_df(
            f"SELECT iata, day, n_dep, avg_dep_delay, n_metar FROM {ctx.namespace}.hub_daily"
        )
    except Exception as e:  # noqa: BLE001
        return False, (f"could not read {ctx.namespace}.hub_daily "
                       f"(right name + columns?): {str(e)[:170]}")
    if got.empty:
        return False, "hub_daily is empty"
    got["day"] = pd.to_datetime(got["day"]).dt.strftime("%Y-%m-%d")
    got["n_dep"] = pd.to_numeric(got["n_dep"], errors="coerce").astype("int64")
    got["n_metar"] = pd.to_numeric(got["n_metar"], errors="coerce").astype("int64")
    got["avg_dep_delay"] = pd.to_numeric(got["avg_dep_delay"], errors="coerce")

    exp = _expected(ctx)
    merged = exp.merge(got, on=["iata", "day"], suffixes=("_e", "_g"), how="outer", indicator=True)
    missing = (merged["_merge"] != "both").sum()
    if missing:
        return False, f"{missing} (iata, day) rows missing or extra vs expected {len(exp)}"
    bad_dep = int((merged["n_dep_e"] != merged["n_dep_g"]).sum())
    bad_met = int((merged["n_metar_e"] != merged["n_metar_g"]).sum())
    bad_delay = int(((merged["avg_dep_delay_e"] - merged["avg_dep_delay_g"]).abs() > 0.02).sum())
    if bad_dep or bad_met or bad_delay:
        return False, (f"value mismatches — n_dep:{bad_dep} "
                       f"n_metar:{bad_met} avg_dep_delay:{bad_delay}")
    return True, ""


def reference(ctx: GradeContext):
    ns = ctx.namespace
    ctx.client.command(
        f"CREATE TABLE {ns}.hub_daily (iata String, day Date, n_dep UInt32, "
        f"avg_dep_delay Float64, n_metar UInt32) ENGINE = MergeTree ORDER BY (iata, day)"
    )
    ctx.client.command(f"""
        INSERT INTO {ns}.hub_daily
        SELECT f.iata, f.day, f.n_dep, f.avg_dep_delay, ifNull(m.n_metar, 0) AS n_metar
        FROM (
            SELECT Origin AS iata, FlightDate AS day, count() AS n_dep,
                   round(avg(DepDelayMinutes), 2) AS avg_dep_delay
            FROM aviation.flights WHERE Origin IN {_IN} GROUP BY Origin, FlightDate
        ) f
        LEFT JOIN (
            SELECT a.iata AS iata, toDate(m.valid_utc) AS day, count() AS n_metar
            FROM aviation.metar m INNER JOIN aviation.airports a ON a.icao = m.station
            GROUP BY a.iata, toDate(m.valid_utc)
        ) m ON f.iata = m.iata AND f.day = m.day
    """)
    return None


PROMPT = """Build a daily operations x weather fact table for the five hub airports
(ATL, ORD, DFW, DEN, LAX) for June 2026.

In your scratch database, create a table named exactly `hub_daily` with these columns:
  - iata           (String)  the hub's IATA code
  - day            (Date)    the calendar day
  - n_dep          (UInt32)  number of departures from that airport that day (all departures)
  - avg_dep_delay  (Float64) average DepDelayMinutes of those departures, rounded to 2 decimals
  - n_metar        (UInt32)  number of METAR observations for that airport that day (0 if none)

One row per (airport, day). Departures come from aviation.flights (Origin is IATA). METARs are in
aviation.metar keyed by ICAO station; map IATA<->ICAO via aviation.airports; bucket m.valid_utc to
the calendar day. That table is the deliverable.
"""

from dsbench.agentic.schema import AgentProblem  # noqa: E402

PROBLEM = AgentProblem(
    id="de_hub_daily", category="de", difficulty="hard",
    title="Daily hub ops x weather fact table", prompt=PROMPT,
    check=check, reference=reference, max_steps=20, tags=("cross-source", "join", "etl"),
)
