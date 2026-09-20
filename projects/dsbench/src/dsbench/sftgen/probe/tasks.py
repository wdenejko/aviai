"""Six held-out probe problems (2 per skill) on fresh domains -- the generalisation test (ADR-004).

Domains are chosen to appear in NEITHER dsbench (aviation) NOR the sftgen generators
(retail/iot/support/payments/web/gym): clinic visits, rideshare, power-grid incidents, marketplace
seller ratings, consumer loans, energy demand. Skills mirror the three fine-tune targets:

  A (dialect date/time):   probe_weekday_visits (ClickHouse toDayOfWeek), probe_utc_peak_trips (tz).
  B (denominator/pop):     probe_incident_share (conditional-null share), probe_pooled_rating.
  C (agentic ML-delivery): probe_loan_default (ROC-AUC), probe_energy_demand (MAE vs median).

Answer-graded problems recompute truth in pandas (check) vs SQL (reference) by DIFFERENT routes, and
table-graded problems grade a held-out metric -- the same oracle discipline as the eval, so
`selftest()` gates them with no model.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from dsbench.agentic.schema import AgentProblem, GradeContext

_UTC_ADD = {"Boston": 4, "Miami": 4, "Houston": 5, "Denver": 6, "Seattle": 7, "Phoenix": 7}
_CITIES = list(_UTC_ADD)


def _ins(ctx: GradeContext, table: str, cols: str, df: pd.DataFrame) -> None:
    ctx.client.command(f"CREATE TABLE {ctx.namespace}.{table} ({cols}) "
                       f"ENGINE = MergeTree ORDER BY tuple()")
    ctx.client.insert_df(f"{ctx.namespace}.{table}", df)


def _num(ans, default=None):
    m = re.findall(r"-?\d+(?:\.\d+)?", "" if ans is None else str(ans))
    return float(m[-1]) if m else default


# ---- A1: weekday numbering (ClickHouse toDayOfWeek is ISO, Tuesday = 2) ----

def _visits_setup(ctx: GradeContext) -> None:
    rng = np.random.default_rng(31)
    n = 3200
    # Date (not DateTime) -> no ClickHouse timezone round-trip; weekday is unambiguous.
    dates = np.datetime64("2025-01-01") + rng.integers(0, 365, n).astype("timedelta64[D]")
    df = pd.DataFrame({"visit_id": np.arange(1, n + 1, dtype="uint32"),
                       "visit_date": pd.to_datetime(dates).date,
                       "clinic": rng.choice(["north", "south", "central"], n)})
    _ins(ctx, "clinic_visits", "visit_id UInt32, visit_date Date, clinic String", df)


def _visits_check(ctx: GradeContext) -> tuple[bool, str]:
    df = ctx.client.query_df(f"SELECT visit_date FROM {ctx.namespace}.clinic_visits")
    truth = int((pd.to_datetime(df["visit_date"]).dt.dayofweek == 1).sum())  # Tuesday
    got = _num(ctx.answer)
    ok = got is not None and int(got) == truth
    return ok, "" if ok else f"expected {truth} Tuesday visits; got {ctx.answer!r}"


def _visits_ref(ctx: GradeContext):
    r = ctx.client.query(f"SELECT count() FROM {ctx.namespace}.clinic_visits "
                         f"WHERE toDayOfWeek(visit_date) = 2")
    return int(r.result_rows[0][0])


PROBE_WEEKDAY = AgentProblem(
    id="probe_weekday_visits", category="da", difficulty="medium",
    title="Clinic visits on a Tuesday", setup=_visits_setup, check=_visits_check,
    reference=_visits_ref, max_steps=8, tags=("probe", "weekday"),
    prompt=("Table `clinic_visits(visit_id, visit_date, clinic)`, one row per visit; visit_date is "
            "the date. How many visits happened on a Tuesday? Reply with ONLY the integer."),
)


# ---- A2: timezone direction (local -> UTC) ----

def _trips_setup(ctx: GradeContext) -> None:
    rng = np.random.default_rng(32)
    n = 4000
    # Store the LOCAL hour directly (UInt8) -> no DateTime/timezone round-trip ambiguity.
    df = pd.DataFrame({"trip_id": np.arange(1, n + 1, dtype="uint32"),
                       "request_hour": rng.integers(0, 24, n).astype("uint8"),
                       "city": rng.choice(_CITIES, n)})
    _ins(ctx, "rideshare_trips", "trip_id UInt32, request_hour UInt8, city String", df)


def _trips_check(ctx: GradeContext) -> tuple[bool, str]:
    df = ctx.client.query_df(f"SELECT request_hour, city FROM {ctx.namespace}.rideshare_trips")
    utc_h = (df["request_hour"].astype(int) + df["city"].map(_UTC_ADD)) % 24
    truth = int(utc_h.value_counts().idxmax())
    got = _num(ctx.answer)
    ok = got is not None and int(got) == truth
    return ok, "" if ok else f"expected UTC hour {truth}; got {ctx.answer!r}"


def _trips_ref(ctx: GradeContext):
    r = ctx.client.query(
        f"SELECT (request_hour + multiIf(city='Boston',4, city='Miami',4, city='Houston',5, "
        f"city='Denver',6, 7)) % 24 AS uh FROM {ctx.namespace}.rideshare_trips "
        f"GROUP BY uh ORDER BY count() DESC LIMIT 1")
    return int(r.result_rows[0][0])


PROBE_UTC = AgentProblem(
    id="probe_utc_peak_trips", category="da", difficulty="hard",
    title="Busiest UTC hour of rideshare requests", setup=_trips_setup, check=_trips_check,
    reference=_trips_ref, max_steps=10, tags=("probe", "timezone"),
    prompt=("Table `rideshare_trips(trip_id, request_hour, city)`. request_hour is the LOCAL "
            "clock-hour (0-23) in its city. Cities and their offset to UTC: Boston=UTC-4, "
            "Miami=UTC-4, Houston=UTC-5, Denver=UTC-6, Seattle=UTC-7, Phoenix=UTC-7. Converting "
            "each request to UTC, which UTC clock-hour (0-23) has the most requests? Reply with "
            "ONLY the integer hour."),
)


# ---- B1: conditional-null share ----

def _grid_setup(ctx: GradeContext) -> None:
    rng = np.random.default_rng(33)
    n = 4000
    major = rng.random(n) < 0.3
    cols = {"incident_id": np.arange(1, n + 1, dtype="uint32"),
            "region": rng.choice(["west", "east", "north"], n), "major": major.astype("int64")}
    for c in ["root_a", "root_b", "root_c", "root_d", "root_e"]:
        v = np.round(rng.gamma(2.0, 25.0, n), 1)
        v[~major] = np.nan
        cols[c] = v
    # Nullable so non-major rows are NULL (skipped by sum), not NaN (which ClickHouse propagates).
    _ins(ctx, "grid_incidents",
         "incident_id UInt32, region String, major UInt8, root_a Nullable(Float64), "
         "root_b Nullable(Float64), root_c Nullable(Float64), root_d Nullable(Float64), "
         "root_e Nullable(Float64)", pd.DataFrame(cols))


def _grid_check(ctx: GradeContext) -> tuple[bool, str]:
    roots = ["root_a", "root_b", "root_c", "root_d", "root_e"]
    df = ctx.client.query_df(f"SELECT {', '.join(roots)} FROM {ctx.namespace}.grid_incidents")
    filled = df.fillna(0.0)
    truth = round(100.0 * float(filled["root_b"].sum()) / float(filled.to_numpy().sum()), 1)
    got = _num(ctx.answer)
    ok = got is not None and abs(got - truth) <= 0.2
    return ok, "" if ok else f"expected ~{truth}%; got {ctx.answer!r}"


def _grid_ref(ctx: GradeContext):
    den = " + ".join(f"sum({c})" for c in ["root_a", "root_b", "root_c", "root_d", "root_e"])
    r = ctx.client.query(f"SELECT 100.0 * sum(root_b) / ({den}) "
                         f"FROM {ctx.namespace}.grid_incidents")
    return round(float(r.result_rows[0][0]), 1)


PROBE_SHARE = AgentProblem(
    id="probe_incident_share", category="da", difficulty="hard",
    title="Share of grid outage minutes from root cause B", setup=_grid_setup, check=_grid_check,
    reference=_grid_ref, max_steps=10, tags=("probe", "denominator"),
    prompt=("Table `grid_incidents(incident_id, region, major, root_a..root_e)`. The five root_* "
            "columns hold minutes and are populated only for major incidents (major = 1), else "
            "null. Of the total minutes across all five root_* columns, what percentage is root_b? "
            "Reply rounded to 1 decimal."),
)


# ---- B2: pooled vs mean-of-means ----

def _ratings_setup(ctx: GradeContext) -> None:
    rng = np.random.default_rng(34)
    sellers = [f"s{i}" for i in range(7)]
    counts = rng.integers(30, 900, len(sellers))
    rates = rng.uniform(0.5, 0.97, len(sellers))
    rows = []
    for s, c, p in zip(sellers, counts, rates, strict=True):
        pos = (rng.random(c) < p).astype("int64")
        rows.append(pd.DataFrame({"seller": s, "positive": pos}))
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "rating_id", np.arange(1, len(df) + 1, dtype="uint32"))
    _ins(ctx, "seller_ratings", "rating_id UInt32, seller String, positive UInt8", df)


def _ratings_check(ctx: GradeContext) -> tuple[bool, str]:
    df = ctx.client.query_df(f"SELECT positive FROM {ctx.namespace}.seller_ratings")
    truth = round(float(df["positive"].mean()), 4)
    got = _num(ctx.answer)
    ok = got is not None and abs(got - truth) <= 0.0005
    return ok, "" if ok else f"expected {truth}; got {ctx.answer!r}"


def _ratings_ref(ctx: GradeContext):
    r = ctx.client.query(f"SELECT sum(positive) / count() FROM {ctx.namespace}.seller_ratings")
    return round(float(r.result_rows[0][0]), 4)


PROBE_POOLED = AgentProblem(
    id="probe_pooled_rating", category="da", difficulty="medium",
    title="Overall positive rating rate", setup=_ratings_setup, check=_ratings_check,
    reference=_ratings_ref, max_steps=8, tags=("probe", "pooled"),
    prompt=("Table `seller_ratings(rating_id, seller, positive)` where positive is 1/0 and sellers "
            "have very different rating counts. What is the overall positive rate across ALL "
            "ratings (total positives / total ratings)? Reply as a fraction to 4 decimals."),
)


# ---- C1: loan default (classification, ROC-AUC) ----

def _split(n: int, rng: np.random.Generator, frac: float = 0.2) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[rng.permutation(n)[: int(n * frac)]] = True
    return m


def _loan_setup(ctx: GradeContext) -> None:
    rng = np.random.default_rng(35)
    n = 4000
    amt = rng.uniform(1, 40, n)
    term = rng.choice([12, 24, 36, 60], n)
    income = rng.gamma(3.0, 20.0, n)
    grade = rng.choice(["A", "B", "C", "D"], n)
    g_eff = np.select([grade == "A", grade == "B"], [-0.8, -0.2], default=0.7)
    logit = 0.05 * amt - 0.03 * income + 0.01 * term + g_eff + rng.normal(0, 0.7, n)
    default = (logit > 0).astype("int64")
    df = pd.DataFrame({"id": np.arange(1, n + 1, dtype="uint32"), "amt": amt, "term": term,
                       "income": income, "grade": grade, "default": default})
    te = _split(n, rng)
    cols = "id UInt32, amt Float64, term UInt16, income Float64, grade String, default UInt8"
    _ins(ctx, "loan_train", cols, df[~te])
    _ins(ctx, "loan_test", "id UInt32, amt Float64, term UInt16, income Float64, grade String",
         df[te].drop(columns=["default"]))
    _ins(ctx, "loan_test_key", "id UInt32, default UInt8", df[te][["id", "default"]])


def _loan_check(ctx: GradeContext) -> tuple[bool, str]:
    from sklearn.metrics import roc_auc_score
    key = ctx.client.query_df(f"SELECT id, default FROM {ctx.namespace}.loan_test_key")
    try:
        pred = ctx.client.query_df(f"SELECT id, score FROM {ctx.namespace}.loan_pred")
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.loan_pred (id, score): {str(e)[:120]}"
    m = key.merge(pred, on="id", how="left")
    if m["score"].isna().any():
        return False, f"{int(m['score'].isna().sum())} test rows have no prediction"
    auc = roc_auc_score(m["default"], m["score"])
    return (auc >= 0.65), (f"ROC-AUC {auc:.3f}" if auc >= 0.65 else f"ROC-AUC {auc:.3f} < 0.65")


def _loan_ref(ctx: GradeContext):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder
    tr = ctx.client.query_df(f"SELECT amt, term, income, grade, default FROM "
                             f"{ctx.namespace}.loan_train")
    te = ctx.client.query_df(f"SELECT id, amt, term, income, grade FROM "
                             f"{ctx.namespace}.loan_test ORDER BY id")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    xtr = tr[["amt", "term", "income"]].copy()
    xtr["grade"] = enc.fit_transform(tr[["grade"]])
    xte = te[["amt", "term", "income"]].copy()
    xte["grade"] = enc.transform(te[["grade"]])
    clf = HistGradientBoostingClassifier(random_state=0).fit(xtr, tr["default"])
    out = pd.DataFrame({"id": te["id"].astype("uint32"), "score": clf.predict_proba(xte)[:, 1]})
    ctx.client.command(f"CREATE TABLE {ctx.namespace}.loan_pred (id UInt32, score Float64) "
                       f"ENGINE = MergeTree ORDER BY id")
    ctx.client.insert_df(f"{ctx.namespace}.loan_pred", out)
    return None


PROBE_LOAN = AgentProblem(
    id="probe_loan_default", category="ds", difficulty="hard",
    title="Predict loan default (ROC-AUC)", setup=_loan_setup, check=_loan_check,
    reference=_loan_ref, max_steps=24, tags=("probe", "ml", "classification"),
    prompt=("Two tables are loaded: `loan_train(id, amt, term, income, grade, default)` (default "
            "is 1/0) and `loan_test(id, amt, term, income, grade)` with no default. Fit a "
            "classifier, score every test loan, and leave a table `loan_pred` with columns `id` "
            "(UInt32) and `score` (Float64) = P(default) for all test rows. Graded by ROC-AUC; "
            "reach >= 0.65."),
)


# ---- C2: energy demand (regression, MAE vs median) ----

def _energy_setup(ctx: GradeContext) -> None:
    rng = np.random.default_rng(36)
    n = 4000
    temp = rng.uniform(-5, 35, n)
    hour = rng.integers(0, 24, n)
    dow = rng.integers(0, 7, n)
    region = rng.choice(["r1", "r2", "r3"], n)
    r_eff = np.select([region == "r1", region == "r2"], [30.0, -10.0], default=0.0)
    demand = (200 + 3.0 * np.abs(temp - 18) + 4.0 * hour + 6.0 * (dow >= 5) + r_eff
              + rng.normal(0, 15, n)).round(1)
    df = pd.DataFrame({"id": np.arange(1, n + 1, dtype="uint32"), "temp": temp,
                       "hour": hour.astype("uint16"), "dow": dow.astype("uint8"),
                       "region": region, "demand": demand})
    te = _split(n, rng)
    cols = "id UInt32, temp Float64, hour UInt16, dow UInt8, region String, demand Float64"
    _ins(ctx, "energy_train", cols, df[~te])
    _ins(ctx, "energy_test", "id UInt32, temp Float64, hour UInt16, dow UInt8, region String",
         df[te].drop(columns=["demand"]))
    _ins(ctx, "energy_test_key", "id UInt32, demand Float64", df[te][["id", "demand"]])


def _energy_check(ctx: GradeContext) -> tuple[bool, str]:
    from sklearn.metrics import mean_absolute_error
    key = ctx.client.query_df(f"SELECT id, demand FROM {ctx.namespace}.energy_test_key")
    try:
        pred = ctx.client.query_df(f"SELECT id, pred FROM {ctx.namespace}.energy_pred")
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.energy_pred (id, pred): {str(e)[:120]}"
    m = key.merge(pred, on="id", how="left")
    if m["pred"].isna().any():
        return False, f"{int(m['pred'].isna().sum())} test rows have no prediction"
    tr = ctx.client.query_df(f"SELECT demand FROM {ctx.namespace}.energy_train")
    base = mean_absolute_error(m["demand"], np.full(len(m), float(np.median(tr["demand"]))))
    mae = mean_absolute_error(m["demand"], m["pred"].astype(float))
    ok = mae <= base * 0.94
    lift = 100.0 * (base - mae) / base
    return ok, (f"MAE {mae:.2f} (lift {lift:.1f}%)" if ok
                else f"MAE {mae:.2f} vs median {base:.2f} (lift {lift:.1f}% < 6%)")


def _energy_ref(ctx: GradeContext):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import OrdinalEncoder
    tr = ctx.client.query_df(f"SELECT temp, hour, dow, region, demand FROM "
                             f"{ctx.namespace}.energy_train")
    te = ctx.client.query_df(f"SELECT id, temp, hour, dow, region FROM "
                             f"{ctx.namespace}.energy_test ORDER BY id")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    xtr = tr[["temp", "hour", "dow"]].copy()
    xtr["region"] = enc.fit_transform(tr[["region"]])
    xte = te[["temp", "hour", "dow"]].copy()
    xte["region"] = enc.transform(te[["region"]])
    reg = HistGradientBoostingRegressor(random_state=0).fit(xtr, tr["demand"])
    out = pd.DataFrame({"id": te["id"].astype("uint32"), "pred": reg.predict(xte)})
    ctx.client.command(f"CREATE TABLE {ctx.namespace}.energy_pred (id UInt32, pred Float64) "
                       f"ENGINE = MergeTree ORDER BY id")
    ctx.client.insert_df(f"{ctx.namespace}.energy_pred", out)
    return None


PROBE_ENERGY = AgentProblem(
    id="probe_energy_demand", category="ds", difficulty="hard",
    title="Predict energy demand (MAE)", setup=_energy_setup, check=_energy_check,
    reference=_energy_ref, max_steps=24, tags=("probe", "ml", "regression"),
    prompt=("Two tables are loaded: `energy_train(id, temp, hour, dow, region, demand)` and "
            "`energy_test(id, temp, hour, dow, region)` with no demand. Fit a regressor, estimate "
            "demand for every test row, and leave a table `energy_pred` with columns `id` (UInt32) "
            "and `pred` (Float64). Graded by mean absolute error; beat the constant train-median "
            "estimate by at least 6%."),
)


PROBE_TASKS = [PROBE_WEEKDAY, PROBE_UTC, PROBE_SHARE, PROBE_POOLED, PROBE_LOAN, PROBE_ENERGY]


def selftest() -> int:
    """Oracle gate: setup -> reference -> check per probe task, no model. Returns the fail count."""
    from dsbench.agentic.ch import get_client
    fails = 0
    for p in PROBE_TASKS:
        ns = f"probe_selftest_{p.id}"
        admin = get_client(database="default")
        admin.command(f"DROP DATABASE IF EXISTS {ns}")
        admin.command(f"CREATE DATABASE {ns}")
        ctx = GradeContext(client=get_client(database=ns), namespace=ns)
        try:
            p.setup(ctx)
            ans = p.reference(ctx)
            if ans is not None:
                ctx.answer = str(ans)  # answer-graded tasks: feed the reference value to check
            ok, reason = p.check(ctx)
            print(f"{'PASS' if ok else 'FAIL'} {p.id}: {reason or 'ok'}")
            fails += 0 if ok else 1
        finally:
            get_client(database="default").command(f"DROP DATABASE IF EXISTS {ns}")
    return fails


if __name__ == "__main__":
    raise SystemExit(selftest())
