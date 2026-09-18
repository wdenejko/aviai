"""DS (setup + table-graded ML): predict flight cancellations -- an IMBALANCED classification.

Only ~2% of hub flights cancel, so accuracy is a trap: predicting "never cancels" scores ~0.98 yet
is useless. The honest task is to RANK flights by cancellation risk from pre-departure features
(schedule/carrier/route/distance -- no post-hoc columns), scored by ROC-AUC on a held-out set. The
agent writes a cancellation PROBABILITY per test row; check() re-reads the held-out labels and
computes AUC from the actual scores. The reference (ordinal-encoded gradient boosting) hits ~0.87,
well above the 0.75 bar and far above the 0.50 a majority-class / constant predictor would score.

Determinism: a fixed `cityHash64` bucket splits train/test; the test id is materialised once so the
agent's view and the grader's key align without a natural key (see ds_delay_predict).
"""
from __future__ import annotations

from dsbench.agentic.schema import AgentProblem, GradeContext

THRESHOLD = 0.75  # reference AUC ~0.87; a majority-class / constant predictor is 0.5
_HUBS = "('ATL','ORD','DFW','DEN','LAX')"
_BUCKET = ("cityHash64(FlightDate, Reporting_Airline, Flight_Number_Reporting_Airline, "
           "Origin, Dest, CRSDepTime) % 100")
_POOL = (
    f"(SELECT FlightDate, Reporting_Airline, Flight_Number_Reporting_Airline AS fnum, Origin, "
    f"Dest, CRSDepTime, Distance, Cancelled, {_BUCKET} AS b FROM aviation.flights "
    f"WHERE Origin IN {_HUBS})"
)
_COLS = ("FlightDate AS fl_date, Reporting_Airline AS carrier, Origin AS origin, Dest AS dest, "
         "CRSDepTime AS crs_dep_time, Distance AS distance")


def setup(ctx: GradeContext) -> None:
    ns = ctx.namespace
    ctx.client.command(
        f"CREATE TABLE {ns}.cancel_train ENGINE = MergeTree ORDER BY tuple() AS "
        f"SELECT {_COLS}, toUInt8(Cancelled) AS cancelled FROM {_POOL} WHERE b >= 8 AND b < 40"
    )
    ctx.client.command(
        f"CREATE TABLE {ns}.cancel_test_full ENGINE = MergeTree ORDER BY id AS "
        f"SELECT row_number() OVER (ORDER BY FlightDate, Reporting_Airline, fnum, Origin, Dest, "
        f"CRSDepTime) AS id, {_COLS}, toUInt8(Cancelled) AS cancelled FROM {_POOL} WHERE b < 8"
    )
    ctx.client.command(
        f"CREATE TABLE {ns}.cancel_test ENGINE = MergeTree ORDER BY id AS "
        f"SELECT id, fl_date, carrier, origin, dest, crs_dep_time, distance "
        f"FROM {ns}.cancel_test_full"
    )
    ctx.client.command(
        f"CREATE TABLE {ns}.cancel_test_key ENGINE = MergeTree ORDER BY id AS "
        f"SELECT id, cancelled FROM {ns}.cancel_test_full"
    )


def check(ctx: GradeContext) -> tuple[bool, str]:
    from sklearn.metrics import roc_auc_score

    truth = ctx.client.query_df(f"SELECT id, cancelled FROM {ctx.namespace}.cancel_test_key")
    try:
        pred = ctx.client.query_df(f"SELECT id, score FROM {ctx.namespace}.cancel_pred")
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.cancel_pred (id, score): {str(e)[:150]}"
    m = truth.merge(pred, on="id", how="left")
    if m["score"].isna().any():
        return False, f"{int(m['score'].isna().sum())} of {len(truth)} test rows have no score"
    try:
        auc = float(roc_auc_score(m["cancelled"].astype(int), m["score"].astype(float)))
    except (ValueError, TypeError) as e:
        return False, f"could not score predictions as ROC-AUC: {str(e)[:120]}"
    ok = auc >= THRESHOLD
    return ok, "" if ok else f"test ROC-AUC {auc:.3f} < {THRESHOLD} (random = 0.5)"


def reference(ctx: GradeContext):
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder

    tr = ctx.client.query_df(
        f"SELECT carrier, origin, dest, crs_dep_time, fl_date, distance, cancelled "
        f"FROM {ctx.namespace}.cancel_train"
    )
    te = ctx.client.query_df(
        f"SELECT id, carrier, origin, dest, crs_dep_time, fl_date, distance "
        f"FROM {ctx.namespace}.cancel_test ORDER BY id"
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
    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(xtr, tr["cancelled"].astype(int))
    scores = clf.predict_proba(xte)[:, 1]
    out = pd.DataFrame({"id": te["id"].astype("uint32"), "score": scores})
    ctx.client.command(
        f"CREATE TABLE {ctx.namespace}.cancel_pred (id UInt32, score Float64) "
        f"ENGINE = MergeTree ORDER BY id"
    )
    ctx.client.insert_df(f"{ctx.namespace}.cancel_pred", out)
    return None


PROMPT = """Your scratch database has two tables built from June 2026 hub flights (ATL, ORD, DFW,
DEN, LAX):
  - `cancel_train(fl_date Date, carrier String, origin String, dest String, crs_dep_time String,
     distance Float64, cancelled UInt8)` -- the labelled training set; `cancelled` is 1 if the
     flight was cancelled, else 0.
  - `cancel_test(id UInt32, fl_date, carrier, origin, dest, crs_dep_time, distance)` -- the test
     flights, WITHOUT the label.

All features are known before departure (schedule/carrier/route/distance); crs_dep_time is local
"hhmm". Cancellations are RARE (~2%), so accuracy is meaningless -- predicting "never
cancels" scores ~0.98. Train a model on cancel_train and, for EVERY row of cancel_test, predict the
PROBABILITY of cancellation. Write your predictions to a table named exactly
`cancel_pred(id UInt32, score Float64)` in your scratch database -- one row per test id. `score` is
the predicted probability of cancellation (higher = more likely cancelled).

Grading is by ranking quality (ROC-AUC) on the held-out labels, so a probability/score is required;
you must clearly beat random (0.5). Use run_python (pandas + scikit-learn are available). That table
is the deliverable.
"""

PROBLEM = AgentProblem(
    id="ds_cancel_predict", category="ds", difficulty="hard",
    title="Predict flight cancellations (imbalanced, ROC-AUC)", prompt=PROMPT,
    check=check, reference=reference, setup=setup, max_steps=24,
    tags=("ml", "classification", "imbalanced", "auc"),
)
