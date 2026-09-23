"""Synthetic, non-aviation ML tasks for Target C -- the sandbox problems the teacher agent solves.

Target C teaches the WORKFLOW the models fail on (ADR-004): train a model, predict EVERY test row,
write the exact-schema output table, then stop. So the tasks mirror the `ds_*` mechanics (labelled
train + unlabelled test + a hidden key; grade by a held-out metric with a beat-the-baseline bar) but
on SYNTHETIC, non-aviation data -- decontaminated by construction and licence-free.

Each task carries setup/check/reference (the AgentProblem contract) so it runs in the ADR-003
sandbox and is oracle-gated: `selftest()` runs setup -> reference -> check with NO model, proving
the task is solvable and the bar is reachable before any teacher time is spent (mirrors
dsbench-agent-selftest).

The data has a real learnable signal (a GBM clears the bar; a constant/degenerate answer fails), so
a kept trajectory reflects genuine ML delivery, not a lucky guess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsbench.agentic.schema import AgentProblem, GradeContext

# Generic agent context -- NO aviation schema (that would self-contaminate Target C trajectories).
SYSTEM_C = """You are a data scientist operating a sandbox to complete a machine-learning task.
Tools: run_sql (one ClickHouse statement against YOUR scratch database, the default), run_python
(a container with pandas/numpy/scikit-learn/clickhouse_connect; the CLICKHOUSE_* env points at your
scratch database, so a client built from it reads/writes there by default), finish (end the task).

Work step by step: inspect the tables, train a model on the training table, predict for EVERY row of
the test table, and WRITE your predictions to the exact table name and schema the task specifies —
that table is the deliverable. Then call finish. You have a limited number of steps; be efficient
and do not over-tune. To write predictions from run_python, create the table then insert, e.g.
`client.command('CREATE TABLE t (id UInt32, pred Float64) ENGINE=MergeTree ORDER BY id')` then
`client.insert_df('t', df)`."""

# Generic tool schemas (no aviation mention), same executors as the measurement harness.
TOOL_SCHEMAS_C = [
    {"type": "function", "function": {
        "name": "run_sql",
        "description": ("Run ONE ClickHouse SQL statement against your scratch database (the "
                        "default). SELECT/SHOW return rows (capped); DDL/DML return 'OK'."),
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": ("Execute Python 3 in the analysis container (pandas, numpy, scikit-learn, "
                        "clickhouse_connect). CLICKHOUSE_* env points at your scratch DB. "
                        "Returns stdout+stderr; print what you want to see."),
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Call when the deliverable table is built and complete.",
        "parameters": {"type": "object", "properties": {"answer": {"description": "optional"}}}}},
]

_LINES = ["A", "B", "C"]
_CARRIERS = ["north", "south", "east", "west"]
_PLANS = ["basic", "pro", "enterprise"]
_SOURCES = ["web", "email", "phone"]
_QUEUES = ["billing", "network", "hardware", "account"]
_HIST = ["thin", "fair", "good"]
_REGIONS = ["emea", "amer", "apac"]
_KINDS = ["renewal", "addon", "onetime"]


def _run_seed(ctx: GradeContext, base: int) -> int:
    """Vary the synthetic data per run so repetitions are DISTINCT datasets, not the same one.

    The C generator names each run's scratch DB `sftc_<task>_<run_ix>`, so the trailing integer is
    the run index; a volume run of N reps then trains on N different datasets. selftest() uses a
    non-numeric namespace and falls back to the base seed (a fixed, oracle-gated dataset).
    """
    try:
        return base + int(ctx.namespace.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return base


def _split(n: int, rng: np.random.Generator, test_frac: float = 0.2) -> np.ndarray:
    """Return a boolean is_test mask over 0..n-1 (deterministic given rng)."""
    is_test = np.zeros(n, dtype=bool)
    idx = rng.permutation(n)[: int(n * test_frac)]
    is_test[idx] = True
    return is_test


def _insert(ctx: GradeContext, table: str, ddl_cols: str, df: pd.DataFrame) -> None:
    ctx.client.command(f"CREATE TABLE {ctx.namespace}.{table} ({ddl_cols}) "
                       f"ENGINE = MergeTree ORDER BY id")
    ctx.client.insert_df(f"{ctx.namespace}.{table}", df)


# ---- Task 1: widget defect (binary classification, graded ROC-AUC) ----

def _widget_data(seed: int, n: int = 4000):
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    line = rng.choice(_LINES, size=n)
    line_eff = np.select([line == "A", line == "B"], [0.6, -0.4], default=0.1)
    logit = 0.9 * x1 - 0.8 * x2 + 0.5 * x3 + line_eff + rng.normal(scale=0.7, size=n) - 0.3
    label = (logit > 0).astype("int64")
    df = pd.DataFrame({"id": np.arange(1, n + 1, dtype="uint32"),
                       "x1": x1, "x2": x2, "x3": x3, "line": line, "label": label})
    return df, _split(n, rng)


def _widget_setup(ctx: GradeContext) -> None:
    df, is_test = _widget_data(_run_seed(ctx, 101))
    tr, te = df[~is_test], df[is_test]
    _insert(ctx, "widget_train", "id UInt32, x1 Float64, x2 Float64, x3 Float64, line String, "
                                 "label UInt8", tr)
    _insert(ctx, "widget_test", "id UInt32, x1 Float64, x2 Float64, x3 Float64, line String",
            te.drop(columns=["label"]))
    _insert(ctx, "widget_test_key", "id UInt32, label UInt8", te[["id", "label"]])


def _widget_check(ctx: GradeContext) -> tuple[bool, str]:
    from sklearn.metrics import roc_auc_score
    key = ctx.client.query_df(f"SELECT id, label FROM {ctx.namespace}.widget_test_key")
    try:
        pred = ctx.client.query_df(f"SELECT id, score FROM {ctx.namespace}.defect_pred")
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.defect_pred (id, score): {str(e)[:120]}"
    m = key.merge(pred, on="id", how="left")
    if m["score"].isna().any():
        return False, f"{int(m['score'].isna().sum())} of {len(key)} test rows have no prediction"
    auc = roc_auc_score(m["label"], m["score"])
    return (auc >= 0.65), (f"ROC-AUC {auc:.3f}" if auc >= 0.65 else f"ROC-AUC {auc:.3f} < 0.65")


def _widget_reference(ctx: GradeContext):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder
    tr = ctx.client.query_df(f"SELECT x1, x2, x3, line, label FROM {ctx.namespace}.widget_train")
    te = ctx.client.query_df(f"SELECT id, x1, x2, x3, line FROM {ctx.namespace}.widget_test "
                             f"ORDER BY id")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    xtr = tr[["x1", "x2", "x3"]].copy()
    xtr["line"] = enc.fit_transform(tr[["line"]])
    xte = te[["x1", "x2", "x3"]].copy()
    xte["line"] = enc.transform(te[["line"]])
    clf = HistGradientBoostingClassifier(random_state=0).fit(xtr, tr["label"])
    out = pd.DataFrame({"id": te["id"].astype("uint32"),
                        "score": clf.predict_proba(xte)[:, 1]})
    ctx.client.command(f"CREATE TABLE {ctx.namespace}.defect_pred (id UInt32, score Float64) "
                       f"ENGINE = MergeTree ORDER BY id")
    ctx.client.insert_df(f"{ctx.namespace}.defect_pred", out)
    return None


_WIDGET_PROMPT = """Two tables are already loaded for you:
  - `widget_train(id, x1, x2, x3, line, label)` — labelled; label is 1 (defective) or 0.
  - `widget_test(id, x1, x2, x3, line)` — the widgets to score, with no label.

Fit a classifier on the training widgets, then score every test widget. Deliver the result as a
table called `defect_pred` holding just two columns: `id` (UInt32, matching each test widget) and
`score` (Float64, the predicted probability the widget is defective). Cover all of them. It is
graded by ROC-AUC against the withheld labels and must reach 0.65 or better. scikit-learn is
available via run_python.
"""

WIDGET = AgentProblem(
    id="mlc_widget_defect", category="ds", difficulty="hard",
    title="Predict widget defect (ROC-AUC)",
    prompt=_WIDGET_PROMPT, check=_widget_check, reference=_widget_reference, setup=_widget_setup,
    max_steps=24, tags=("ml", "classification", "delivery"),
)


# ---- Task 2: delivery time (regression, graded MAE vs median baseline) ----

def _delivery_data(seed: int, n: int = 4000):
    rng = np.random.default_rng(seed)
    dist = rng.uniform(1, 50, size=n)
    weight = rng.gamma(2.0, 2.0, size=n)
    hour = rng.integers(0, 24, size=n)
    carrier = rng.choice(_CARRIERS, size=n)
    car_eff = np.select([carrier == "north", carrier == "south"], [8.0, -5.0], default=0.0)
    minutes = (20 + 0.8 * dist + 2.5 * weight + 0.6 * hour + car_eff
               + rng.normal(scale=6.0, size=n)).round(1)
    df = pd.DataFrame({"id": np.arange(1, n + 1, dtype="uint32"), "dist": dist, "weight": weight,
                       "hour": hour.astype("uint16"), "carrier": carrier, "minutes": minutes})
    return df, _split(n, rng)


def _delivery_setup(ctx: GradeContext) -> None:
    df, is_test = _delivery_data(_run_seed(ctx, 202))
    tr, te = df[~is_test], df[is_test]
    cols = "id UInt32, dist Float64, weight Float64, hour UInt16, carrier String, minutes Float64"
    _insert(ctx, "delivery_train", cols, tr)
    _insert(ctx, "delivery_test", "id UInt32, dist Float64, weight Float64, hour UInt16, "
                                  "carrier String", te.drop(columns=["minutes"]))
    _insert(ctx, "delivery_test_key", "id UInt32, minutes Float64", te[["id", "minutes"]])


def _delivery_check(ctx: GradeContext) -> tuple[bool, str]:
    import numpy as np
    from sklearn.metrics import mean_absolute_error
    key = ctx.client.query_df(f"SELECT id, minutes FROM {ctx.namespace}.delivery_test_key")
    try:
        pred = ctx.client.query_df(f"SELECT id, pred FROM {ctx.namespace}.delivery_pred")
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.delivery_pred (id, pred): {str(e)[:120]}"
    m = key.merge(pred, on="id", how="left")
    if m["pred"].isna().any():
        return False, f"{int(m['pred'].isna().sum())} of {len(key)} test rows have no prediction"
    tr = ctx.client.query_df(f"SELECT minutes FROM {ctx.namespace}.delivery_train")
    base = mean_absolute_error(m["minutes"], np.full(len(m), float(np.median(tr["minutes"]))))
    mae = mean_absolute_error(m["minutes"], m["pred"].astype(float))
    ok = mae <= base * 0.94  # beat predict-the-median by >= 6%
    lift = 100.0 * (base - mae) / base
    return ok, (f"MAE {mae:.2f} (lift {lift:.1f}%)" if ok else
                f"MAE {mae:.2f} vs median {base:.2f} (lift {lift:.1f}% < 6%)")


def _delivery_reference(ctx: GradeContext):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import OrdinalEncoder
    tr = ctx.client.query_df(f"SELECT dist, weight, hour, carrier, minutes "
                             f"FROM {ctx.namespace}.delivery_train")
    te = ctx.client.query_df(f"SELECT id, dist, weight, hour, carrier "
                             f"FROM {ctx.namespace}.delivery_test ORDER BY id")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    xtr = tr[["dist", "weight", "hour"]].copy()
    xtr["carrier"] = enc.fit_transform(tr[["carrier"]])
    xte = te[["dist", "weight", "hour"]].copy()
    xte["carrier"] = enc.transform(te[["carrier"]])
    reg = HistGradientBoostingRegressor(random_state=0).fit(xtr, tr["minutes"])
    out = pd.DataFrame({"id": te["id"].astype("uint32"), "pred": reg.predict(xte)})
    ctx.client.command(f"CREATE TABLE {ctx.namespace}.delivery_pred (id UInt32, pred Float64) "
                       f"ENGINE = MergeTree ORDER BY id")
    ctx.client.insert_df(f"{ctx.namespace}.delivery_pred", out)
    return None


_DELIVERY_PROMPT = """Two tables are already loaded for you:
  - `delivery_train(id, dist, weight, hour, carrier, minutes)` — labelled; minutes is delivery time.
  - `delivery_test(id, dist, weight, hour, carrier)` — the deliveries to estimate, with no minutes.

Fit a regressor on the training deliveries, then estimate minutes for every test delivery. Deliver
the result as a table called `delivery_pred` with two columns: `id` (UInt32, matching each test
delivery) and `pred` (Float64, the estimated minutes). Cover all of them. Grading uses mean absolute
error, and you must beat the constant training-median estimate by at least 6%. Use run_python.
"""

DELIVERY = AgentProblem(
    id="mlc_delivery_time", category="ds", difficulty="hard", title="Predict delivery time (MAE)",
    prompt=_DELIVERY_PROMPT, check=_delivery_check, reference=_delivery_reference,
    setup=_delivery_setup, max_steps=24, tags=("ml", "regression", "delivery"),
)


# ---- Task 3: churn (RARE-event classification, graded average precision) ----
# Why a separate task: task 1 is balanced, so accuracy-shaped thinking works there. Here the
# positive rate is ~5%, so the majority answer ("nobody churns") scores at the base rate and fails.
# It teaches ranking under imbalance -- a distinct delivery skill from plain binary classification.

def _churn_data(seed: int, n: int = 6000):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, size=n).astype("float64")
    spend = rng.gamma(3.0, 20.0, size=n)
    tickets = rng.poisson(0.8, size=n).astype("float64")
    plan = rng.choice(_PLANS, size=n)
    plan_eff = np.select([plan == "basic", plan == "pro"], [0.9, -0.7], default=0.0)
    logit = (-2.3 - 0.030 * tenure + 0.95 * tickets - 0.013 * spend + plan_eff
             + rng.normal(scale=0.6, size=n))
    label = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype("int64")
    df = pd.DataFrame({"id": np.arange(1, n + 1, dtype="uint32"), "tenure": tenure,
                       "spend": spend, "tickets": tickets, "plan": plan, "label": label})
    return df, _split(n, rng)


def _churn_setup(ctx: GradeContext) -> None:
    df, is_test = _churn_data(_run_seed(ctx, 303))
    tr, te = df[~is_test], df[is_test]
    cols = "id UInt32, tenure Float64, spend Float64, tickets Float64, plan String"
    _insert(ctx, "churn_train", cols + ", label UInt8", tr)
    _insert(ctx, "churn_test", cols, te.drop(columns=["label"]))
    _insert(ctx, "churn_test_key", "id UInt32, label UInt8", te[["id", "label"]])


def _churn_check(ctx: GradeContext) -> tuple[bool, str]:
    from sklearn.metrics import average_precision_score
    key = ctx.client.query_df(f"SELECT id, label FROM {ctx.namespace}.churn_test_key")
    try:
        pred = ctx.client.query_df(f"SELECT id, score FROM {ctx.namespace}.churn_pred")
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.churn_pred (id, score): {str(e)[:120]}"
    m = key.merge(pred, on="id", how="left")
    if m["score"].isna().any():
        return False, f"{int(m['score'].isna().sum())} of {len(key)} test rows have no prediction"
    ap = average_precision_score(m["label"], m["score"].astype(float))
    rate = float(m["label"].mean())
    ok = ap >= 0.15
    return ok, (f"AP {ap:.3f} (base rate {rate:.3f})" if ok else
                f"AP {ap:.3f} < 0.15 (base rate {rate:.3f})")


def _churn_reference(ctx: GradeContext):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder
    tr = ctx.client.query_df(f"SELECT tenure, spend, tickets, plan, label "
                             f"FROM {ctx.namespace}.churn_train")
    te = ctx.client.query_df(f"SELECT id, tenure, spend, tickets, plan "
                             f"FROM {ctx.namespace}.churn_test ORDER BY id")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    xtr = tr[["tenure", "spend", "tickets"]].copy()
    xtr["plan"] = enc.fit_transform(tr[["plan"]])
    xte = te[["tenure", "spend", "tickets"]].copy()
    xte["plan"] = enc.transform(te[["plan"]])
    clf = HistGradientBoostingClassifier(random_state=0).fit(xtr, tr["label"])
    out = pd.DataFrame({"id": te["id"].astype("uint32"),
                        "score": clf.predict_proba(xte)[:, 1]})
    ctx.client.command(f"CREATE TABLE {ctx.namespace}.churn_pred (id UInt32, score Float64) "
                       f"ENGINE = MergeTree ORDER BY id")
    ctx.client.insert_df(f"{ctx.namespace}.churn_pred", out)
    return None


_CHURN_PROMPT = """Two tables are already loaded for you:
  - `churn_train(id, tenure, spend, tickets, plan, label)` — labelled; label is 1 if the customer
    churned. Churn is RARE, so most rows are 0.
  - `churn_test(id, tenure, spend, tickets, plan)` — the customers to score, with no label.

Fit a model on the training customers, then score every test customer. Deliver a table called
`churn_pred` with two columns: `id` (UInt32) and `score` (Float64, the predicted probability the
customer churns). Cover all of them. Grading uses average precision (area under the
precision-recall curve) and must reach 0.15 or better — note that predicting a single constant for
everyone scores at the base rate and will not pass. scikit-learn is available via run_python.
"""

CHURN = AgentProblem(
    id="mlc_churn_rare", category="ds", difficulty="hard",
    title="Rank rare churn (average precision)",
    prompt=_CHURN_PROMPT, check=_churn_check, reference=_churn_reference, setup=_churn_setup,
    max_steps=24, tags=("ml", "classification", "imbalanced", "delivery"),
)


# ---- Task 4: energy load (TEMPORAL split regression, graded vs a naive forecast) ----
# Why a separate task: tasks 1-3 split rows at random, so shuffling is harmless. Here the test rows
# are the FUTURE -- the last 20% of the hours. It teaches the agent to respect a time boundary and
# to model seasonality instead of carrying the recent level forward.

def _energy_data(seed: int, n: int = 5000):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    hour = (t % 24).astype("int64")
    dow = ((t // 24) % 7).astype("int64")
    temp = 12.0 + 9.0 * np.sin(2 * np.pi * (t - 2000) / (24 * 365)) + rng.normal(scale=2.5, size=n)
    load = (120.0 + 20.0 * np.sin(2 * np.pi * (hour - 9) / 24) + 8.0 * np.cos(2 * np.pi * hour / 12)
            + np.where(dow >= 5, -7.0, 0.0) + 1.3 * temp + 0.004 * t
            + rng.normal(scale=3.0, size=n)).round(2)
    df = pd.DataFrame({"id": np.arange(1, n + 1, dtype="uint32"), "ts_hour": t.astype("uint32"),
                       "hour": hour.astype("uint16"), "dow": dow.astype("uint16"),
                       "temp": temp, "load_mw": load})
    is_test = t >= int(n * 0.8)  # TEMPORAL split: the tail is the future, never a random sample
    return df, is_test


def _energy_setup(ctx: GradeContext) -> None:
    df, is_test = _energy_data(_run_seed(ctx, 404))
    tr, te = df[~is_test], df[is_test]
    cols = "id UInt32, ts_hour UInt32, hour UInt16, dow UInt16, temp Float64"
    _insert(ctx, "energy_train", cols + ", load_mw Float64", tr)
    _insert(ctx, "energy_test", cols, te.drop(columns=["load_mw"]))
    _insert(ctx, "energy_test_key", "id UInt32, load_mw Float64", te[["id", "load_mw"]])


def _energy_check(ctx: GradeContext) -> tuple[bool, str]:
    from sklearn.metrics import mean_absolute_error
    key = ctx.client.query_df(f"SELECT id, load_mw FROM {ctx.namespace}.energy_test_key")
    try:
        pred = ctx.client.query_df(f"SELECT id, pred FROM {ctx.namespace}.load_pred")
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.load_pred (id, pred): {str(e)[:120]}"
    m = key.merge(pred, on="id", how="left")
    if m["pred"].isna().any():
        return False, f"{int(m['pred'].isna().sum())} of {len(key)} test rows have no prediction"
    # Naive forecast: carry the mean of the last 24 training hours forward over the whole horizon.
    tail = ctx.client.query_df(f"SELECT load_mw FROM {ctx.namespace}.energy_train "
                               f"ORDER BY ts_hour DESC LIMIT 24")
    base = mean_absolute_error(m["load_mw"], np.full(len(m), float(tail["load_mw"].mean())))
    mae = mean_absolute_error(m["load_mw"], m["pred"].astype(float))
    lift = 100.0 * (base - mae) / base
    ok = mae <= base * 0.75  # beat the carry-forward forecast by >= 25%
    return ok, (f"MAE {mae:.2f} vs naive {base:.2f} (lift {lift:.1f}%)" if ok else
                f"MAE {mae:.2f} vs naive {base:.2f} (lift {lift:.1f}% < 25%)")


def _energy_reference(ctx: GradeContext):
    from sklearn.ensemble import HistGradientBoostingRegressor
    tr = ctx.client.query_df(f"SELECT hour, dow, temp, ts_hour, load_mw "
                             f"FROM {ctx.namespace}.energy_train")
    te = ctx.client.query_df(f"SELECT id, hour, dow, temp, ts_hour "
                             f"FROM {ctx.namespace}.energy_test ORDER BY id")
    feats = ["hour", "dow", "temp", "ts_hour"]
    reg = HistGradientBoostingRegressor(random_state=0).fit(tr[feats], tr["load_mw"])
    out = pd.DataFrame({"id": te["id"].astype("uint32"), "pred": reg.predict(te[feats])})
    ctx.client.command(f"CREATE TABLE {ctx.namespace}.load_pred (id UInt32, pred Float64) "
                       f"ENGINE = MergeTree ORDER BY id")
    ctx.client.insert_df(f"{ctx.namespace}.load_pred", out)
    return None


_ENERGY_PROMPT = """Two tables of hourly grid readings are already loaded for you:
  - `energy_train(id, ts_hour, hour, dow, temp, load_mw)` — the history. `ts_hour` counts hours from
    the start of the record, `hour` is the hour of day, `dow` the day of week (5 and 6 are weekend).
  - `energy_test(id, ts_hour, hour, dow, temp)` — the hours to forecast, with no `load_mw`.

The test hours are the FUTURE: every `ts_hour` in the test table comes after every `ts_hour` in the
training table. Fit a model on the history and forecast `load_mw` for every test hour. Deliver a
table called `load_pred` with two columns: `id` (UInt32) and `pred` (Float64). Cover all of them.
Grading uses mean absolute error and you must beat the naive forecast — carrying the average of the
last 24 training hours forward — by at least 25%. Use run_python.
"""

ENERGY = AgentProblem(
    id="mlc_energy_load", category="ds", difficulty="hard",
    title="Forecast hourly load (temporal MAE)",
    prompt=_ENERGY_PROMPT, check=_energy_check, reference=_energy_reference, setup=_energy_setup,
    max_steps=24, tags=("ml", "regression", "timeseries", "delivery"),
)


# ---- Task 5: ticket routing (MULTICLASS, graded macro-F1, STRING deliverable) ----
# Why a separate task: every task above delivers a Float64 score. Here the deliverable is a
# categorical LABEL, and macro-F1 weights all four queues equally, so collapsing onto the most
# common queue scores ~0.1. It teaches exact-schema categorical output and per-class coverage.

def _ticket_data(seed: int, n: int = 5000):
    rng = np.random.default_rng(seed)
    words = rng.gamma(2.0, 30.0, size=n)
    attachments = rng.poisson(0.6, size=n).astype("float64")
    prior = rng.integers(0, 12, size=n).astype("float64")
    src = rng.choice(_SOURCES, size=n)
    src_bill = np.where(src == "web", 0.7, -0.3)
    # The four intercepts are set so the utilities have comparable MEANS: macro-F1 weights every
    # queue equally, so a lopsided generator starves one class and caps the achievable score no
    # matter how good the model is. The slopes carry the signal; the intercepts carry the balance.
    util = np.stack([
        1.0 + 0.022 * words - 1.10 * attachments + src_bill,         # billing
        -1.2 - 0.010 * words + 0.62 * prior,                         # network
        -0.2 + 0.014 * words + 2.20 * attachments,                   # hardware
        3.0 - 0.020 * words + 0.90 * attachments - 0.10 * prior,     # account
    ], axis=1) + rng.normal(scale=0.45, size=(n, 4))
    queue = np.array(_QUEUES)[util.argmax(axis=1)]
    df = pd.DataFrame({"id": np.arange(1, n + 1, dtype="uint32"), "words": words,
                       "attachments": attachments, "prior": prior, "src": src, "queue": queue})
    return df, _split(n, rng)


def _ticket_setup(ctx: GradeContext) -> None:
    df, is_test = _ticket_data(_run_seed(ctx, 505))
    tr, te = df[~is_test], df[is_test]
    cols = "id UInt32, words Float64, attachments Float64, prior Float64, src String"
    _insert(ctx, "ticket_train", cols + ", queue String", tr)
    _insert(ctx, "ticket_test", cols, te.drop(columns=["queue"]))
    _insert(ctx, "ticket_test_key", "id UInt32, queue String", te[["id", "queue"]])


def _ticket_check(ctx: GradeContext) -> tuple[bool, str]:
    from sklearn.metrics import f1_score
    key = ctx.client.query_df(f"SELECT id, queue FROM {ctx.namespace}.ticket_test_key")
    try:
        pred = ctx.client.query_df(f"SELECT id, queue AS pq FROM {ctx.namespace}.route_pred")
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.route_pred (id, queue): {str(e)[:120]}"
    m = key.merge(pred, on="id", how="left")
    if m["pq"].isna().any():
        return False, f"{int(m['pq'].isna().sum())} of {len(key)} test rows have no prediction"
    bad = sorted(set(m["pq"].astype(str)) - set(_QUEUES))
    if bad:
        return False, f"predicted queues outside the allowed set: {bad[:4]}"
    f1 = f1_score(m["queue"], m["pq"].astype(str), average="macro", labels=_QUEUES, zero_division=0)
    return (f1 >= 0.50), (f"macro-F1 {f1:.3f}" if f1 >= 0.50 else f"macro-F1 {f1:.3f} < 0.50")


def _ticket_reference(ctx: GradeContext):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder
    tr = ctx.client.query_df(f"SELECT words, attachments, prior, src, queue "
                             f"FROM {ctx.namespace}.ticket_train")
    te = ctx.client.query_df(f"SELECT id, words, attachments, prior, src "
                             f"FROM {ctx.namespace}.ticket_test ORDER BY id")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    xtr = tr[["words", "attachments", "prior"]].copy()
    xtr["src"] = enc.fit_transform(tr[["src"]])
    xte = te[["words", "attachments", "prior"]].copy()
    xte["src"] = enc.transform(te[["src"]])
    clf = HistGradientBoostingClassifier(random_state=0).fit(xtr, tr["queue"])
    out = pd.DataFrame({"id": te["id"].astype("uint32"), "queue": clf.predict(xte).astype(str)})
    ctx.client.command(f"CREATE TABLE {ctx.namespace}.route_pred (id UInt32, queue String) "
                       f"ENGINE = MergeTree ORDER BY id")
    ctx.client.insert_df(f"{ctx.namespace}.route_pred", out)
    return None


_TICKET_PROMPT = """Two tables are already loaded for you:
  - `ticket_train(id, words, attachments, prior, src, queue)` — labelled; `queue` is the team the
    ticket was routed to, one of billing, network, hardware, account.
  - `ticket_test(id, words, attachments, prior, src)` — the tickets to route, with no queue.

Fit a classifier on the training tickets, then route every test ticket. Deliver a table called
`route_pred` with two columns: `id` (UInt32) and `queue` (String, one of the four queue names
exactly as spelled above). Cover all of them. Grading uses macro-averaged F1 across the four
queues and must reach 0.50 or better, so every queue counts equally no matter how common it is.
scikit-learn is available via run_python.
"""

TICKET = AgentProblem(
    id="mlc_ticket_route", category="ds", difficulty="hard",
    title="Route tickets to queues (macro-F1)",
    prompt=_TICKET_PROMPT, check=_ticket_check, reference=_ticket_reference, setup=_ticket_setup,
    max_steps=24, tags=("ml", "multiclass", "delivery"),
)


# ---- Task 6: credit default (LEAKAGE TRAP, graded ROC-AUC) ----
# Why a separate task: this is the discipline test. `collections_flag` is recorded AFTER the
# outcome, so it is identical to the label in training and is empty at predict
# time. An agent that trains on every column gets a beautiful validation score and ~0.5 on the
# real grade;
# only an agent that reads the data dictionary and drops the column passes.

def _credit_data(seed: int, n: int = 5000):
    rng = np.random.default_rng(seed)
    income = rng.gamma(6.0, 9000.0, size=n)
    debt_ratio = rng.beta(2.0, 5.0, size=n)
    age = rng.integers(21, 70, size=n).astype("float64")
    hist = rng.choice(_HIST, size=n)
    hist_eff = np.select([hist == "thin", hist == "good"], [1.7, -1.8], default=0.0)
    logit = (0.6 - 0.000032 * income + 6.0 * debt_ratio - 0.034 * age + hist_eff
             + rng.normal(scale=0.30, size=n))
    label = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype("int64")
    # The leak: collections is opened exactly when a default is confirmed, so on settled rows the
    # flag IS the label. That purity is what makes the trap bite -- a tree splits on it first, the
    # flag=0 node is clean, boosting stops there, and at predict time (every flag 0) the model
    # emits one constant. An imperfect leak would leave the node impure and the model would quietly
    # fall back on the honest features, which teaches nothing.
    flag = label.astype("uint8")
    df = pd.DataFrame({"id": np.arange(1, n + 1, dtype="uint32"), "income": income,
                       "debt_ratio": debt_ratio, "age": age, "hist": hist,
                       "collections_flag": flag, "label": label})
    return df, _split(n, rng)


def _credit_setup(ctx: GradeContext) -> None:
    df, is_test = _credit_data(_run_seed(ctx, 606))
    df = df.copy()
    # Pending applications have no outcome yet, so nothing has been sent to collections.
    df.loc[is_test, "collections_flag"] = np.uint8(0)
    tr, te = df[~is_test], df[is_test]
    cols = ("id UInt32, income Float64, debt_ratio Float64, age Float64, hist String, "
            "collections_flag UInt8")
    _insert(ctx, "credit_train", cols + ", label UInt8", tr)
    _insert(ctx, "credit_test", cols, te.drop(columns=["label"]))
    _insert(ctx, "credit_test_key", "id UInt32, label UInt8", te[["id", "label"]])


def _credit_check(ctx: GradeContext) -> tuple[bool, str]:
    from sklearn.metrics import roc_auc_score
    key = ctx.client.query_df(f"SELECT id, label FROM {ctx.namespace}.credit_test_key")
    try:
        pred = ctx.client.query_df(f"SELECT id, score FROM {ctx.namespace}.default_pred")
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.default_pred (id, score): {str(e)[:120]}"
    m = key.merge(pred, on="id", how="left")
    if m["score"].isna().any():
        return False, f"{int(m['score'].isna().sum())} of {len(key)} test rows have no prediction"
    auc = roc_auc_score(m["label"], m["score"].astype(float))
    return (auc >= 0.70), (f"ROC-AUC {auc:.3f}" if auc >= 0.70 else
                           f"ROC-AUC {auc:.3f} < 0.70 (a leaked feature collapses to ~0.5)")


def _credit_reference(ctx: GradeContext):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder
    # The reference deliberately EXCLUDES collections_flag -- that is the lesson of this task.
    tr = ctx.client.query_df(f"SELECT income, debt_ratio, age, hist, label "
                             f"FROM {ctx.namespace}.credit_train")
    te = ctx.client.query_df(f"SELECT id, income, debt_ratio, age, hist "
                             f"FROM {ctx.namespace}.credit_test ORDER BY id")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    xtr = tr[["income", "debt_ratio", "age"]].copy()
    xtr["hist"] = enc.fit_transform(tr[["hist"]])
    xte = te[["income", "debt_ratio", "age"]].copy()
    xte["hist"] = enc.transform(te[["hist"]])
    clf = HistGradientBoostingClassifier(random_state=0).fit(xtr, tr["label"])
    out = pd.DataFrame({"id": te["id"].astype("uint32"),
                        "score": clf.predict_proba(xte)[:, 1]})
    ctx.client.command(f"CREATE TABLE {ctx.namespace}.default_pred (id UInt32, score Float64) "
                       f"ENGINE = MergeTree ORDER BY id")
    ctx.client.insert_df(f"{ctx.namespace}.default_pred", out)
    return None


_CREDIT_PROMPT = """Two tables are already loaded for you:
  - `credit_train(id, income, debt_ratio, age, hist, collections_flag, label)` — settled
    applications; label is 1 if the borrower defaulted.
  - `credit_test(id, income, debt_ratio, age, hist, collections_flag)` — pending applications to
    score, with no label.

Data dictionary:
  income            applicant annual income, recorded at application time
  debt_ratio        debt-to-income ratio, recorded at application time
  age               applicant age, recorded at application time
  hist              credit history bucket (thin, fair, good), recorded at application time
  collections_flag  set by the collections team AFTER an account is confirmed in default. It is
                    therefore only ever populated for settled applications; for every pending
                    application in `credit_test` it is still 0.

Fit a model on the settled applications and score every pending one. Deliver a table called
`default_pred` with two columns: `id` (UInt32) and `score` (Float64, the predicted probability of
default). Cover all of them. Grading uses ROC-AUC against the eventual outcomes and must reach 0.70
or better. Use run_python.
"""

CREDIT = AgentProblem(
    id="mlc_credit_leak", category="ds", difficulty="hard",
    title="Score default without the leaked column (ROC-AUC)",
    prompt=_CREDIT_PROMPT, check=_credit_check, reference=_credit_reference, setup=_credit_setup,
    max_steps=24, tags=("ml", "classification", "leakage", "delivery"),
)


# ---- Task 7: upsell (MULTI-TABLE aggregation, graded ROC-AUC) ----
# Why a separate task: the signal is not in the customer table at all -- it lives in a long event
# log that has to be aggregated per customer first. Modelling the customer columns alone lands
# around chance. It is the DE/DS crossover: build the features before you fit anything.

def _upsell_data(seed: int, n: int = 2500):
    rng = np.random.default_rng(seed)
    region = rng.choice(_REGIONS, size=n)
    signup_days = rng.integers(30, 900, size=n).astype("float64")
    intensity = rng.gamma(2.0, 2.0, size=n)
    counts = rng.poisson(np.clip(intensity * 3.0, 0.5, None)).astype("int64")
    cust = pd.DataFrame({"id": np.arange(1, n + 1, dtype="uint32"), "region": region,
                         "signup_days": signup_days})
    rows = []
    totals = np.zeros(n)
    recency = np.full(n, 400.0)
    for i in range(n):
        k = int(counts[i])
        if k == 0:
            continue
        amt = rng.gamma(2.0, 18.0 + 6.0 * intensity[i], size=k)
        day = rng.integers(1, int(min(signup_days[i], 365)) + 1, size=k)
        kind = rng.choice(_KINDS, size=k)
        totals[i] = float(amt.sum())
        recency[i] = float(day.min())
        rows.append(pd.DataFrame({"cust_id": np.full(k, i + 1, dtype="uint32"),
                                  "days_ago": day.astype("uint16"), "amount": amt, "kind": kind}))
    events = (pd.concat(rows, ignore_index=True) if rows else
              pd.DataFrame({"cust_id": np.array([], dtype="uint32"),
                            "days_ago": np.array([], dtype="uint16"),
                            "amount": np.array([]), "kind": np.array([], dtype=object)}))
    logit = (-1.1 + 0.0048 * totals + 0.10 * counts - 0.0045 * recency
             + rng.normal(scale=0.5, size=n))
    cust["label"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype("int64")
    return cust, events, _split(n, rng)


def _upsell_setup(ctx: GradeContext) -> None:
    cust, events, is_test = _upsell_data(_run_seed(ctx, 707))
    tr, te = cust[~is_test], cust[is_test]
    cols = "id UInt32, region String, signup_days Float64"
    _insert(ctx, "cust_train", cols + ", label UInt8", tr)
    _insert(ctx, "cust_test", cols, te.drop(columns=["label"]))
    _insert(ctx, "cust_test_key", "id UInt32, label UInt8", te[["id", "label"]])
    # The event log covers BOTH splits -- it is history, not an outcome, so it is legitimately
    # available for the test customers too. It has no `id`, so it is created directly.
    ctx.client.command(f"CREATE TABLE {ctx.namespace}.events (cust_id UInt32, days_ago UInt16, "
                       f"amount Float64, kind String) ENGINE = MergeTree ORDER BY cust_id")
    ctx.client.insert_df(f"{ctx.namespace}.events", events)


def _upsell_check(ctx: GradeContext) -> tuple[bool, str]:
    from sklearn.metrics import roc_auc_score
    key = ctx.client.query_df(f"SELECT id, label FROM {ctx.namespace}.cust_test_key")
    try:
        pred = ctx.client.query_df(f"SELECT id, score FROM {ctx.namespace}.upsell_pred")
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.upsell_pred (id, score): {str(e)[:120]}"
    m = key.merge(pred, on="id", how="left")
    if m["score"].isna().any():
        return False, f"{int(m['score'].isna().sum())} of {len(key)} test rows have no prediction"
    auc = roc_auc_score(m["label"], m["score"].astype(float))
    return (auc >= 0.70), (f"ROC-AUC {auc:.3f}" if auc >= 0.70 else
                           f"ROC-AUC {auc:.3f} < 0.70 (the signal is in `events`)")


def _upsell_features(ctx: GradeContext, table: str, extra: str) -> pd.DataFrame:
    return ctx.client.query_df(f"""
        SELECT c.id AS id, c.region AS region, c.signup_days AS signup_days, {extra}
               coalesce(e.n, 0) AS n, coalesce(e.total, 0) AS total,
               coalesce(e.recency, 400) AS recency
        FROM {ctx.namespace}.{table} AS c
        LEFT JOIN (SELECT cust_id, count() AS n, sum(amount) AS total,
                          min(days_ago) AS recency
                   FROM {ctx.namespace}.events GROUP BY cust_id) AS e ON e.cust_id = c.id
        ORDER BY c.id""")


def _upsell_reference(ctx: GradeContext):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder
    tr = _upsell_features(ctx, "cust_train", "c.label AS label,")
    te = _upsell_features(ctx, "cust_test", "")
    feats = ["signup_days", "n", "total", "recency"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    xtr = tr[feats].astype(float).copy()
    xtr["region"] = enc.fit_transform(tr[["region"]])
    xte = te[feats].astype(float).copy()
    xte["region"] = enc.transform(te[["region"]])
    clf = HistGradientBoostingClassifier(random_state=0).fit(xtr, tr["label"])
    out = pd.DataFrame({"id": te["id"].astype("uint32"),
                        "score": clf.predict_proba(xte)[:, 1]})
    ctx.client.command(f"CREATE TABLE {ctx.namespace}.upsell_pred (id UInt32, score Float64) "
                       f"ENGINE = MergeTree ORDER BY id")
    ctx.client.insert_df(f"{ctx.namespace}.upsell_pred", out)
    return None


_UPSELL_PROMPT = """Three tables are already loaded for you:
  - `cust_train(id, region, signup_days, label)` — labelled; label is 1 if the customer took the
    upsell offer.
  - `cust_test(id, region, signup_days)` — the customers to score, with no label.
  - `events(cust_id, days_ago, amount, kind)` — the purchase log, one row per purchase, covering
    the customers in BOTH tables. A customer with no purchases has no rows here.

The customer columns alone carry very little signal; the useful features have to be built by
aggregating `events` per customer (for example how many purchases, how much in total, and how
recently). Build those features, fit a model on the training customers, then score every test
customer. Deliver a table called `upsell_pred` with two columns: `id` (UInt32) and `score`
(Float64, the predicted probability of taking the offer). Cover all of them — including customers
with no purchases. Grading uses ROC-AUC and must reach 0.70 or better. Use run_python or run_sql.
"""

UPSELL = AgentProblem(
    id="mlc_upsell_join", category="ds", difficulty="hard",
    title="Upsell propensity from an event log (ROC-AUC)",
    prompt=_UPSELL_PROMPT, check=_upsell_check, reference=_upsell_reference, setup=_upsell_setup,
    max_steps=24, tags=("ml", "classification", "multitable", "delivery"),
)


ML_TASKS = [WIDGET, DELIVERY, CHURN, ENERGY, TICKET, CREDIT, UPSELL]


def selftest(reps: int = 1) -> int:
    """Oracle gate: setup -> reference -> check for each task, no model. Returns failure count.

    `reps` runs every task on DIFFERENT datasets (the namespace's trailing index feeds `_run_seed`,
    and rep 0 reproduces the base dataset). One rep only proves a task is solvable on one draw --
    it cannot show that the BAR is reachable in general. Two of these tasks were first written with
    bars the oracle cleared by ~0.01 on the base seed and failed on most others, which would have
    thrown away good teacher trajectories at generation time. Tune against the distribution.
    """
    import re
    import statistics

    from dsbench.agentic.ch import get_client
    fails = 0
    for p in ML_TASKS:
        vals, task_fails = [], 0
        for rep in range(reps):
            ns = f"sftc_selftest_{p.id}_{rep}"
            admin = get_client(database="default")
            admin.command(f"DROP DATABASE IF EXISTS {ns}")
            admin.command(f"CREATE DATABASE {ns}")
            ctx = GradeContext(client=get_client(database=ns), namespace=ns)
            try:
                p.setup(ctx)
                p.reference(ctx)
                ok, reason = p.check(ctx)
                task_fails += 0 if ok else 1
                if reps == 1:
                    print(f"{'PASS' if ok else 'FAIL'} {p.id}: {reason}")
                else:
                    m = re.search(r"(-?\d+\.\d+)", reason)
                    if m:
                        vals.append(float(m.group(1)))
            finally:
                get_client(database="default").command(f"DROP DATABASE IF EXISTS {ns}")
        if reps > 1:
            lo = min(vals) if vals else float("nan")
            med = statistics.median(vals) if vals else float("nan")
            verdict = "PASS" if not task_fails else f"FAIL {task_fails}/{reps}"
            print(f"{verdict:9s} {p.id:22s} min {lo:.3f}  median {med:.3f}", flush=True)
        fails += task_fails
    return fails


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="oracle-gate the synthetic ML tasks")
    ap.add_argument("--reps", type=int, default=1,
                    help="datasets per task; >1 reports the metric distribution")
    raise SystemExit(selftest(reps=ap.parse_args().reps))
