"""DS (setup + table-graded ML): predict taxi-out minutes -- an honest regression.

Unlike arrival delay, TaxiOut is NOT zero-inflated (every departure taxis; median ~18 min), so mean
absolute error is a fair metric and the MAE-optimal constant baseline is the training MEDIAN. The
agent predicts TaxiOut for each held-out flight from pre-departure features; check() computes MAE
against the held-out truth and requires beating the predict-the-median baseline by a clear margin.
The reference (ordinal-encoded gradient boosting) beats it by ~13%, comfortably past the 6% bar,
while any constant prediction is 0% and fails.

Determinism: a fixed `cityHash64` bucket splits train/test; the test id is materialised once so the
agent's view and the grader's key align without a natural key (see ds_delay_predict).
"""
from __future__ import annotations

from dsbench.agentic.schema import AgentProblem, GradeContext

MARGIN = 0.06  # must beat predict-the-median MAE by >=6%; reference beats it by ~13%
_HUBS = "('ATL','ORD','DFW','DEN','LAX')"
_BUCKET = ("cityHash64(FlightDate, Reporting_Airline, Flight_Number_Reporting_Airline, "
           "Origin, Dest, CRSDepTime) % 100")
_POOL = (
    f"(SELECT FlightDate, Reporting_Airline, Flight_Number_Reporting_Airline AS fnum, Origin, "
    f"Dest, CRSDepTime, Distance, TaxiOut, {_BUCKET} AS b FROM aviation.flights "
    f"WHERE Origin IN {_HUBS} AND Cancelled = 0 AND TaxiOut IS NOT NULL)"
)
_COLS = ("FlightDate AS fl_date, Reporting_Airline AS carrier, Origin AS origin, Dest AS dest, "
         "CRSDepTime AS crs_dep_time, Distance AS distance")


def setup(ctx: GradeContext) -> None:
    ns = ctx.namespace
    ctx.client.command(
        f"CREATE TABLE {ns}.taxi_train ENGINE = MergeTree ORDER BY tuple() AS "
        f"SELECT {_COLS}, TaxiOut AS taxi_out FROM {_POOL} WHERE b >= 8 AND b < 40"
    )
    ctx.client.command(
        f"CREATE TABLE {ns}.taxi_test_full ENGINE = MergeTree ORDER BY id AS "
        f"SELECT row_number() OVER (ORDER BY FlightDate, Reporting_Airline, fnum, Origin, Dest, "
        f"CRSDepTime) AS id, {_COLS}, TaxiOut AS taxi_out FROM {_POOL} WHERE b < 8"
    )
    ctx.client.command(
        f"CREATE TABLE {ns}.taxi_test ENGINE = MergeTree ORDER BY id AS "
        f"SELECT id, fl_date, carrier, origin, dest, crs_dep_time, distance "
        f"FROM {ns}.taxi_test_full"
    )
    ctx.client.command(
        f"CREATE TABLE {ns}.taxi_test_key ENGINE = MergeTree ORDER BY id AS "
        f"SELECT id, taxi_out FROM {ns}.taxi_test_full"
    )


def check(ctx: GradeContext) -> tuple[bool, str]:
    import numpy as np
    from sklearn.metrics import mean_absolute_error

    truth = ctx.client.query_df(f"SELECT id, taxi_out FROM {ctx.namespace}.taxi_test_key")
    try:
        pred = ctx.client.query_df(f"SELECT id, pred FROM {ctx.namespace}.taxi_pred")
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.taxi_pred (id, pred): {str(e)[:150]}"
    m = truth.merge(pred, on="id", how="left")
    if m["pred"].isna().any():
        return False, f"{int(m['pred'].isna().sum())} of {len(truth)} test rows have no prediction"
    tr = ctx.client.query_df(f"SELECT taxi_out FROM {ctx.namespace}.taxi_train")
    median = float(np.median(tr["taxi_out"].astype(float)))  # MAE-optimal constant baseline
    y = m["taxi_out"].astype(float)
    base = mean_absolute_error(y, np.full(len(y), median))
    mae = mean_absolute_error(y, m["pred"].astype(float))
    ok = mae <= base * (1.0 - MARGIN)
    lift = 100.0 * (base - mae) / base
    return ok, "" if ok else (f"MAE {mae:.3f} vs predict-median {base:.3f} "
                              f"(lift {lift:.1f}% < {MARGIN * 100:.0f}%)")


def reference(ctx: GradeContext):
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import OrdinalEncoder

    tr = ctx.client.query_df(
        f"SELECT carrier, origin, dest, crs_dep_time, fl_date, distance, taxi_out "
        f"FROM {ctx.namespace}.taxi_train"
    )
    te = ctx.client.query_df(
        f"SELECT id, carrier, origin, dest, crs_dep_time, fl_date, distance "
        f"FROM {ctx.namespace}.taxi_test ORDER BY id"
    )

    def feats(d: pd.DataFrame):
        hour = d["crs_dep_time"].astype(str).str.zfill(4).str[:2].astype(int)
        dow = pd.to_datetime(d["fl_date"]).dt.dayofweek
        return pd.DataFrame({
            "carrier": d["carrier"].astype(str), "origin": d["origin"].astype(str),
            "dest": d["dest"].astype(str), "dep_hour": hour, "dow": dow,
            "distance": d["distance"].astype(float),
        })

    cat = ["carrier", "origin", "dest"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    ftr, fte = feats(tr), feats(te)
    xtr, xte = ftr.copy(), fte.copy()
    xtr[cat] = enc.fit_transform(ftr[cat])
    xte[cat] = enc.transform(fte[cat])
    reg = HistGradientBoostingRegressor(random_state=0)
    reg.fit(xtr, tr["taxi_out"].astype(float))
    out = pd.DataFrame({"id": te["id"].astype("uint32"), "pred": reg.predict(xte)})
    ctx.client.command(
        f"CREATE TABLE {ctx.namespace}.taxi_pred (id UInt32, pred Float64) "
        f"ENGINE = MergeTree ORDER BY id"
    )
    ctx.client.insert_df(f"{ctx.namespace}.taxi_pred", out)
    return None


PROMPT = """Your scratch database has two tables built from June 2026 hub departures (ATL, ORD, DFW,
DEN, LAX), non-cancelled flights with a known taxi-out time:
  - `taxi_train(fl_date Date, carrier String, origin String, dest String, crs_dep_time String,
     distance Float64, taxi_out Float64)` -- the labelled training set; `taxi_out` is the taxi-out
     time in minutes.
  - `taxi_test(id UInt32, fl_date, carrier, origin, dest, crs_dep_time, distance)` -- the test
     flights, WITHOUT the label.

All features are known before departure; crs_dep_time is a local "hhmm" string. Train a regressor on
taxi_train and predict taxi_out (minutes) for EVERY row of taxi_test. Write your predictions to a
table named exactly `taxi_pred(id UInt32, pred Float64)` in your scratch database -- one row per
test id.

Grading is by mean absolute error on the held-out flights; you must beat the trivial baseline of
predicting the training-set median taxi-out for every flight, by a clear margin. Use run_python
(pandas + scikit-learn are available). That table is the deliverable.
"""

PROBLEM = AgentProblem(
    id="ds_taxi_regression", category="ds", difficulty="hard",
    title="Predict taxi-out minutes (MAE)", prompt=PROMPT,
    check=check, reference=reference, setup=setup, max_steps=24,
    tags=("ml", "regression", "mae"),
)
