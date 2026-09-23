"""Negative controls for the synthetic ML tasks -- the other half of the oracle gate.

`ml_tasks.selftest()` proves each task is SOLVABLE (the reference clears the bar). That is only half
of what makes a task worth spending teacher time on. The other half is that the task must be
FAILABLE by the careless approach it is meant to teach against, or a kept trajectory proves nothing
and the target behaviour never gets taught.

This caught a real bug: `mlc_credit_leak` originally leaked an imperfect flag (label AND a 90%
coin). A model trained on every column still scored 0.814 -- the `flag=0` node stayed impure, so the
trees quietly learned the honest features underneath it and the "trap" was decorative. Making the
leak exact collapses the leaky model to 0.501 while the honest reference holds at 0.82+.

Run it the same way as the oracle, against a live sandbox:
    uv run python -m dsbench.sftgen.ml_task_controls
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsbench.agentic.ch import get_client
from dsbench.agentic.schema import AgentProblem, GradeContext
from dsbench.sftgen import ml_tasks as M


def _write(ctx: GradeContext, table: str, df: pd.DataFrame, col: str) -> None:
    ctx.client.command(f"CREATE TABLE {ctx.namespace}.{table} (id UInt32, {col}) "
                       f"ENGINE = MergeTree ORDER BY id")
    ctx.client.insert_df(f"{ctx.namespace}.{table}", df)


def _fit_predict(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str], cat: str, label: str):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    xtr = tr[feats].astype(float).copy()
    xtr[cat] = enc.fit_transform(tr[[cat]])
    xte = te[feats].astype(float).copy()
    xte[cat] = enc.transform(te[[cat]])
    clf = HistGradientBoostingClassifier(random_state=0).fit(xtr, tr[label])
    return clf.predict_proba(xte)[:, 1]


def _leaky_credit(ctx: GradeContext) -> None:
    """Train on EVERY column, including the post-outcome one. The lesson of mlc_credit_leak."""
    tr = ctx.client.query_df(f"SELECT income, debt_ratio, age, hist, collections_flag, label "
                             f"FROM {ctx.namespace}.credit_train")
    te = ctx.client.query_df(f"SELECT id, income, debt_ratio, age, hist, collections_flag "
                             f"FROM {ctx.namespace}.credit_test ORDER BY id")
    feats = ["income", "debt_ratio", "age", "collections_flag"]
    score = _fit_predict(tr, te, feats, "hist", "label")
    _write(ctx, "default_pred", pd.DataFrame({"id": te["id"].astype("uint32"), "score": score}),
           "score Float64")


def _nojoin_upsell(ctx: GradeContext) -> None:
    """Model the customer table alone and never aggregate the event log."""
    tr = ctx.client.query_df(f"SELECT region, signup_days, label FROM {ctx.namespace}.cust_train")
    te = ctx.client.query_df(f"SELECT id, region, signup_days FROM {ctx.namespace}.cust_test "
                             f"ORDER BY id")
    score = _fit_predict(tr, te, ["signup_days"], "region", "label")
    _write(ctx, "upsell_pred", pd.DataFrame({"id": te["id"].astype("uint32"), "score": score}),
           "score Float64")


def _constant_churn(ctx: GradeContext) -> None:
    """One score for everybody -- scores at the base rate under average precision."""
    te = ctx.client.query_df(f"SELECT id FROM {ctx.namespace}.churn_test ORDER BY id")
    _write(ctx, "churn_pred", pd.DataFrame({"id": te["id"].astype("uint32"),
           "score": np.full(len(te), 0.5)}), "score Float64")


def _majority_ticket(ctx: GradeContext) -> None:
    """Route everything to the most common queue -- macro-F1 punishes the collapse."""
    tr = ctx.client.query_df(f"SELECT queue FROM {ctx.namespace}.ticket_train")
    te = ctx.client.query_df(f"SELECT id FROM {ctx.namespace}.ticket_test ORDER BY id")
    top = str(tr["queue"].value_counts().idxmax())
    _write(ctx, "route_pred", pd.DataFrame({"id": te["id"].astype("uint32"),
           "queue": [top] * len(te)}), "queue String")


def _carry_energy(ctx: GradeContext) -> None:
    """Carry the recent level forward instead of modelling the daily shape."""
    tail = ctx.client.query_df(f"SELECT load_mw FROM {ctx.namespace}.energy_train "
                               f"ORDER BY ts_hour DESC LIMIT 24")
    te = ctx.client.query_df(f"SELECT id FROM {ctx.namespace}.energy_test ORDER BY id")
    _write(ctx, "load_pred", pd.DataFrame({"id": te["id"].astype("uint32"),
           "pred": np.full(len(te), float(tail["load_mw"].mean()))}), "pred Float64")


def _median_delivery(ctx: GradeContext) -> None:
    """Predict the training median for every delivery -- exactly the baseline the bar is set on."""
    tr = ctx.client.query_df(f"SELECT minutes FROM {ctx.namespace}.delivery_train")
    te = ctx.client.query_df(f"SELECT id FROM {ctx.namespace}.delivery_test ORDER BY id")
    _write(ctx, "delivery_pred", pd.DataFrame({"id": te["id"].astype("uint32"),
           "pred": np.full(len(te), float(tr["minutes"].median()))}), "pred Float64")


def _coinflip_widget(ctx: GradeContext) -> None:
    """A constant score -- ROC-AUC 0.5, the floor the widget bar sits above."""
    te = ctx.client.query_df(f"SELECT id FROM {ctx.namespace}.widget_test ORDER BY id")
    _write(ctx, "defect_pred", pd.DataFrame({"id": te["id"].astype("uint32"),
           "score": np.full(len(te), 0.5)}), "score Float64")


# task -> (label for the careless approach, builder that produces it)
CONTROLS: dict[str, tuple[str, object]] = {
    "mlc_widget_defect": ("constant score", _coinflip_widget),
    "mlc_delivery_time": ("predict the median", _median_delivery),
    "mlc_churn_rare": ("constant score", _constant_churn),
    "mlc_energy_load": ("carry the level forward", _carry_energy),
    "mlc_ticket_route": ("route to the majority queue", _majority_ticket),
    "mlc_credit_leak": ("train on the leaked column", _leaky_credit),
    "mlc_upsell_join": ("skip the event-log join", _nojoin_upsell),
}


def run_controls(reps: int = 1) -> int:
    """Every control must FAIL its task. Returns the number that wrongly passed."""
    bad = 0
    by_id = {p.id: p for p in M.ML_TASKS}
    for task_id, (label, build) in CONTROLS.items():
        problem: AgentProblem = by_id[task_id]
        for rep in range(reps):
            ns = f"sftc_control_{task_id}_{rep}"
            admin = get_client(database="default")
            admin.command(f"DROP DATABASE IF EXISTS {ns}")
            admin.command(f"CREATE DATABASE {ns}")
            ctx = GradeContext(client=get_client(database=ns), namespace=ns)
            try:
                problem.setup(ctx)
                build(ctx)  # type: ignore[operator]
                passed, reason = problem.check(ctx)
                if passed:
                    bad += 1
                    print(f"BAD  {task_id:22s} [{label}] PASSED -- the task does not "
                          f"discriminate: {reason}", flush=True)
                elif rep == 0:
                    print(f"ok   {task_id:22s} [{label}] fails as intended: {reason}", flush=True)
            finally:
                get_client(database="default").command(f"DROP DATABASE IF EXISTS {ns}")
    return bad


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="negative-control gate for the synthetic ML tasks")
    ap.add_argument("--reps", type=int, default=1, help="datasets per control")
    raise SystemExit(run_controls(reps=ap.parse_args().reps))
