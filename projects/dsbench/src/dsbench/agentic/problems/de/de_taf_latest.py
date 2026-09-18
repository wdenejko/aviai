"""DE (state-graded, argMax dedup): the latest-issued TAF bulletin per hub.

A canonical DE dedup pattern — pick one row per group by a max of another column — plus the
ICAO->IATA join and a period-count sub-aggregate. reference uses ClickHouse
`argMax(product_id, issued)`; check recomputes with a pandas idxmax over `issued` (a different
route), then compares product_id + period count per hub exactly.
"""
from __future__ import annotations

from dsbench.agentic.schema import AgentProblem, GradeContext

_HUBS = ("ATL", "ORD", "DFW", "DEN", "LAX")


def _truth(ctx: GradeContext):
    taf = ctx.client.query_df("SELECT station, product_id, issued FROM aviation.taf")
    air = ctx.client.query_df("SELECT iata, icao FROM aviation.airports")
    latest = taf.loc[taf.groupby("station")["issued"].idxmax(), ["station", "product_id"]]
    latest = latest.rename(columns={"product_id": "latest_pid"})
    n_per = taf.groupby("product_id").size()
    latest["n_periods"] = latest["latest_pid"].map(n_per).astype(int)
    t = latest.merge(air, left_on="station", right_on="icao")
    t = t[t["iata"].isin(_HUBS)]
    return t[["iata", "latest_pid", "n_periods"]].sort_values("iata").reset_index(drop=True)


def check(ctx: GradeContext) -> tuple[bool, str]:
    truth = _truth(ctx)
    try:
        got = ctx.client.query_df(
            f"SELECT iata, latest_pid, n_periods FROM {ctx.namespace}.taf_latest"
        )
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.taf_latest: {str(e)[:150]}"
    if len(got) != len(truth):
        return False, f"expected {len(truth)} rows (one per hub), got {len(got)}"
    m = truth.merge(got, on="iata", how="left", suffixes=("_t", ""))
    bad = m[(m["latest_pid"] != m["latest_pid_t"]) | (m["n_periods"] != m["n_periods_t"])]
    if len(bad):
        r = bad.iloc[0]
        return False, (f"{r['iata']}: expected {r['latest_pid_t']}/{r['n_periods_t']} periods, "
                       f"got {r['latest_pid']}/{r['n_periods']}")
    return True, ""


def reference(ctx: GradeContext):
    ctx.client.command(
        f"CREATE TABLE {ctx.namespace}.taf_latest ENGINE = MergeTree ORDER BY iata AS "
        f"SELECT a.iata AS iata, t.latest_pid AS latest_pid, p.n_periods AS n_periods "
        f"FROM (SELECT station, argMax(product_id, issued) AS latest_pid FROM aviation.taf "
        f"      GROUP BY station) AS t "
        f"JOIN aviation.airports AS a ON a.icao = t.station "
        f"JOIN (SELECT product_id, count() AS n_periods FROM aviation.taf "
        f"GROUP BY product_id) AS p "
        f"     ON p.product_id = t.latest_pid "
        f"WHERE a.iata IN ('ATL','ORD','DFW','DEN','LAX')"
    )
    return None


PROMPT = """In your scratch database, create a table named exactly `taf_latest` with columns:
  - iata         (String)  hub IATA code
  - latest_pid   (String)  product_id of the LATEST-issued TAF bulletin for that hub's station
  - n_periods    (UInt32)  count of ALL rows in aviation.taf carrying that product_id

For each of the five hubs (ATL, ORD, DFW, DEN, LAX): TAFs are in aviation.taf keyed by ICAO
`station`; map IATA<->ICAO via aviation.airports. A bulletin is a `product_id` spanning multiple
period rows; the "latest-issued" bulletin for a station is the one with the maximum `issued`
timestamp. One row per hub. That table is the deliverable.
"""

PROBLEM = AgentProblem(
    id="de_taf_latest", category="de", difficulty="hard",
    title="Latest-issued TAF bulletin per hub", prompt=PROMPT,
    check=check, reference=reference, max_steps=16, tags=("dedup", "argmax", "join"),
)
