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
    df, is_test = _widget_data(101)
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
    df, is_test = _delivery_data(202)
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


ML_TASKS = [WIDGET, DELIVERY]


def selftest() -> int:
    """Oracle gate: setup -> reference -> check for each task, no model. Returns failure count."""
    from dsbench.agentic.ch import get_client
    fails = 0
    for p in ML_TASKS:
        ns = f"sftc_selftest_{p.id}"
        admin = get_client(database="default")
        admin.command(f"DROP DATABASE IF EXISTS {ns}")
        admin.command(f"CREATE DATABASE {ns}")
        ctx = GradeContext(client=get_client(database=ns), namespace=ns)
        try:
            p.setup(ctx)
            p.reference(ctx)
            ok, reason = p.check(ctx)
            print(f"{'PASS' if ok else 'FAIL'} {p.id}: {reason}")
            fails += 0 if ok else 1
        finally:
            get_client(database="default").command(f"DROP DATABASE IF EXISTS {ns}")
    return fails


if __name__ == "__main__":
    raise SystemExit(selftest())
