"""DS (setup + table-graded ML): predict which hub departures arrive late, scored by ROC-AUC.

A genuine, leakage-safe ML task. setup() seeds a labelled train table and an UNLABELLED test table,
both carrying only PRE-DEPARTURE features (schedule + carrier + route + distance) -- none of the
post-hoc columns (DepDelay, ArrTime, taxi/wheels, cause fields) that would leak the outcome. The
agent fits a classifier and writes a delay PROBABILITY per test row.

Why AUC, not accuracy or MAE: the target is zero-inflated (most flights arrive on time), so MAE is
minimised by predicting ~0 (a trivial model "wins") and accuracy barely moves off the majority
class. The real, learnable signal is RANKING -- which flights are likelier to be late -- which
ROC-AUC measures honestly. check() re-reads the held-out labels and computes AUC from the agent's
actual scores (no self-report); the reference (an ordinal-encoded gradient boosting classifier)
clears ~0.72, well above the 0.65 bar and far above the 0.5 of any constant predictor.

Determinism: the train/test split is a fixed `cityHash64` bucket of each flight's identifying
columns, so it is identical every run; the test `id` is materialised once in setup and both the
agent's view and the grader's key derive from that one table, so ids align without a natural key.
"""
from __future__ import annotations

from dsbench.agentic.schema import AgentProblem, GradeContext

THRESHOLD = 0.65  # reference AUC ~0.72; a constant/degenerate predictor is 0.5
_HUBS = "('ATL','ORD','DFW','DEN','LAX')"
# each flight's stable bucket 0..99 from its identifying columns
_BUCKET = ("cityHash64(FlightDate, Reporting_Airline, Flight_Number_Reporting_Airline, "
           "Origin, Dest, CRSDepTime) % 100")
_POOL = (
    f"(SELECT FlightDate, Reporting_Airline, Flight_Number_Reporting_Airline AS fnum, Origin, "
    f"Dest, CRSDepTime, Distance, ArrDelayMinutes, {_BUCKET} AS b FROM aviation.flights "
    f"WHERE Cancelled = 0 AND ArrDelayMinutes IS NOT NULL AND Origin IN {_HUBS})"
)
_COLS = ("FlightDate AS fl_date, Reporting_Airline AS carrier, Origin AS origin, Dest AS dest, "
         "CRSDepTime AS crs_dep_time, Distance AS distance")


def setup(ctx: GradeContext) -> None:
    ns = ctx.namespace
    ctx.client.command(
        f"CREATE TABLE {ns}.delay_train ENGINE = MergeTree ORDER BY tuple() AS "
        f"SELECT {_COLS}, toUInt8(ArrDelayMinutes > 15) AS is_delayed "
        f"FROM {_POOL} WHERE b >= 8 AND b < 40"
    )
    # materialise the test set ONCE (id + features + label), then split into the agent's view
    # (no label) and the grader's key (id + label) so their ids are guaranteed to line up.
    ctx.client.command(
        f"CREATE TABLE {ns}.delay_test_full ENGINE = MergeTree ORDER BY id AS "
        f"SELECT row_number() OVER (ORDER BY FlightDate, Reporting_Airline, fnum, Origin, Dest, "
        f"CRSDepTime) AS id, {_COLS}, toUInt8(ArrDelayMinutes > 15) AS is_delayed "
        f"FROM {_POOL} WHERE b < 8"
    )
    ctx.client.command(
        f"CREATE TABLE {ns}.delay_test ENGINE = MergeTree ORDER BY id AS "
        f"SELECT id, fl_date, carrier, origin, dest, crs_dep_time, distance "
        f"FROM {ns}.delay_test_full"
    )
    ctx.client.command(
        f"CREATE TABLE {ns}.delay_test_key ENGINE = MergeTree ORDER BY id AS "
        f"SELECT id, is_delayed FROM {ns}.delay_test_full"
    )


def check(ctx: GradeContext) -> tuple[bool, str]:
    from sklearn.metrics import roc_auc_score

    truth = ctx.client.query_df(f"SELECT id, is_delayed FROM {ctx.namespace}.delay_test_key")
    try:
        pred = ctx.client.query_df(f"SELECT id, score FROM {ctx.namespace}.delay_pred")
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.delay_pred (id, score): {str(e)[:150]}"
    m = truth.merge(pred, on="id", how="left")
    if m["score"].isna().any():
        return False, f"{int(m['score'].isna().sum())} of {len(truth)} test rows have no score"
    try:
        auc = float(roc_auc_score(m["is_delayed"].astype(int), m["score"].astype(float)))
    except (ValueError, TypeError) as e:
        return False, f"could not score predictions as ROC-AUC: {str(e)[:120]}"
    ok = auc >= THRESHOLD
    return ok, "" if ok else f"test ROC-AUC {auc:.3f} < {THRESHOLD} (random = 0.5)"


def reference(ctx: GradeContext):
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder

    tr = ctx.client.query_df(
        f"SELECT carrier, origin, dest, crs_dep_time, fl_date, distance, is_delayed "
        f"FROM {ctx.namespace}.delay_train"
    )
    te = ctx.client.query_df(
        f"SELECT id, carrier, origin, dest, crs_dep_time, fl_date, distance "
        f"FROM {ctx.namespace}.delay_test ORDER BY id"
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
    xtr = ftr.copy()
    xtr[cat] = enc.fit_transform(ftr[cat])
    xte = fte.copy()
    xte[cat] = enc.transform(fte[cat])
    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(xtr, tr["is_delayed"].astype(int))
    scores = clf.predict_proba(xte)[:, 1]
    out = pd.DataFrame({"id": te["id"].astype("uint32"), "score": scores})
    ctx.client.command(
        f"CREATE TABLE {ctx.namespace}.delay_pred (id UInt32, score Float64) "
        f"ENGINE = MergeTree ORDER BY id"
    )
    ctx.client.insert_df(f"{ctx.namespace}.delay_pred", out)
    return None


PROMPT = """Your scratch database has two tables built from June 2026 hub departures (ATL, ORD, DFW,
DEN, LAX), non-cancelled flights only:
  - `delay_train(fl_date Date, carrier String, origin String, dest String, crs_dep_time String,
     distance Float64, is_delayed UInt8)` -- the labelled training set. `is_delayed` is 1 if the
     flight arrived more than 15 minutes late (ArrDelayMinutes > 15), else 0.
  - `delay_test(id UInt32, fl_date, carrier, origin, dest, crs_dep_time, distance)` -- the test
     flights, WITHOUT the label.

All features are known before departure (schedule, carrier, route, distance). crs_dep_time is a
local "hhmm" string. Train a classifier on delay_train and, for EVERY row of delay_test, predict the
PROBABILITY that the flight is delayed. Write your predictions to a table named exactly
`delay_pred(id UInt32, score Float64)` in your scratch database -- one row per test id. `score` is
the predicted probability of delay (higher = more likely late).

Grading is by ranking quality (ROC-AUC) on the held-out labels, so a probability/score is required,
not just a 0/1 label; you must clearly beat random (0.5). Use run_python (pandas + scikit-learn are
available). That table is the deliverable.
"""

PROBLEM = AgentProblem(
    id="ds_delay_predict", category="ds", difficulty="hard",
    title="Predict late hub arrivals (ROC-AUC)", prompt=PROMPT,
    check=check, reference=reference, setup=setup, max_steps=24,
    tags=("ml", "classification", "auc", "leakage-safe"),
)
