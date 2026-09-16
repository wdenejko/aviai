"""DE (table-graded): per-hub TAF issuance summary.

Trap: a TAF bulletin spans MANY rows (one per forecast period), so the number of bulletins is
count(DISTINCT product_id), not row count — the exact confusion the aviation.taf schema warns about.
Also needs the ICAO->IATA join (taf.station is ICAO). check() recomputes independently in pandas.
"""
from __future__ import annotations

from dsbench.agentic.schema import AgentProblem, GradeContext


def _expected(ctx: GradeContext):
    return ctx.client.query_df(
        "SELECT a.iata AS iata, uniqExact(t.product_id) AS n_bulletins, count() AS n_periods "
        "FROM aviation.taf t INNER JOIN aviation.airports a ON a.icao = t.station "
        "GROUP BY a.iata"
    )


def check(ctx: GradeContext) -> tuple[bool, str]:
    import pandas as pd

    try:
        got = ctx.client.query_df(
            f"SELECT iata, n_bulletins, n_periods FROM {ctx.namespace}.taf_summary"
        )
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.taf_summary: {str(e)[:140]}"
    exp = _expected(ctx)
    for col in ("n_bulletins", "n_periods"):
        got[col] = pd.to_numeric(got[col], errors="coerce").astype("int64")
        exp[col] = exp[col].astype("int64")
    m = exp.merge(got, on="iata", suffixes=("_e", "_g"), how="outer", indicator=True)
    if (m["_merge"] != "both").any():
        return False, f"iata rows mismatch vs expected {sorted(exp['iata'])}"
    bad_b = int((m["n_bulletins_e"] != m["n_bulletins_g"]).sum())
    bad_p = int((m["n_periods_e"] != m["n_periods_g"]).sum())
    if bad_b or bad_p:
        return False, f"value mismatches — n_bulletins:{bad_b} n_periods:{bad_p}"
    return True, ""


def reference(ctx: GradeContext):
    ns = ctx.namespace
    ctx.client.command(
        f"CREATE TABLE {ns}.taf_summary (iata String, n_bulletins UInt32, n_periods UInt32) "
        f"ENGINE = MergeTree ORDER BY iata"
    )
    ctx.client.command(
        f"INSERT INTO {ns}.taf_summary SELECT a.iata, uniqExact(t.product_id), count() "
        f"FROM aviation.taf t INNER JOIN aviation.airports a ON a.icao = t.station GROUP BY a.iata"
    )
    return None


PROMPT = """In your scratch database, create a table named exactly `taf_summary` with columns:
  - iata          (String)  hub IATA code
  - n_bulletins   (UInt32)  number of DISTINCT TAF bulletins issued for that airport in June 2026
  - n_periods     (UInt32)  number of TAF forecast-period rows for that airport

One row per hub airport. TAFs are in aviation.taf keyed by ICAO `station`; a bulletin is identified
by `product_id` and spans multiple rows. Map ICAO->IATA via aviation.airports. Then call finish.
"""

PROBLEM = AgentProblem(
    id="de_taf_summary", category="de", difficulty="medium",
    title="Per-hub TAF issuance summary", prompt=PROMPT,
    check=check, reference=reference, max_steps=10, tags=("taf", "distinct", "join"),
)
