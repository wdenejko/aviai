"""DS (setup + table-graded): train a NOTAM text classifier, predict the held-out test set.

A genuine ML task in the agent loop: featurize NOTAM text, fit a 13-class classifier on the train
split, predict every test row, and write predictions. Grading is honest and execution-verified:
setup() hands the agent an UNLABELED test table; check() re-derives the true labels and computes
accuracy from the agent's actual predictions (not a self-reported number), requiring materially
better than the test majority-class baseline (~0.18). No self-report to game.

Keying: the test id is `row_number() OVER (ORDER BY text, label_id)`. Rows that tie on that order
share the same text AND category (category is a pure function of label_id), so accuracy is invariant
to how ties are numbered — setup and check agree without a unique key.
"""
from __future__ import annotations

from dsbench.agentic.schema import AgentProblem, GradeContext

THRESHOLD = 0.55  # test majority-class baseline ~0.18; a decent tfidf+linear model clears 0.55
_TEST_IDS = ("SELECT row_number() OVER (ORDER BY text, label_id) AS id, {cols} "
             "FROM aviation.notam WHERE split = 'test'")


def setup(ctx: GradeContext) -> None:
    ctx.client.command(
        f"CREATE TABLE {ctx.namespace}.notam_test ENGINE = MergeTree ORDER BY id AS "
        + _TEST_IDS.format(cols="text")
    )


def check(ctx: GradeContext) -> tuple[bool, str]:
    truth = ctx.client.query_df(_TEST_IDS.format(cols="category"))
    try:
        pred = ctx.client.query_df(f"SELECT id, predicted FROM {ctx.namespace}.notam_pred")
    except Exception as e:  # noqa: BLE001
        return False, f"could not read {ctx.namespace}.notam_pred (id, predicted): {str(e)[:150]}"
    m = truth.merge(pred, on="id", how="left")
    if m["predicted"].isna().any():
        missing = int(m["predicted"].isna().sum())
        return False, f"{missing} of {len(truth)} test rows have no prediction"
    acc = float((m["category"].astype(str) == m["predicted"].astype(str)).mean())
    ok = acc >= THRESHOLD
    return ok, "" if ok else f"test accuracy {acc:.3f} < {THRESHOLD} (test majority baseline ~0.18)"


def reference(ctx: GradeContext):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    tr = ctx.client.query_df("SELECT text, category FROM aviation.notam WHERE split = 'train'")
    te = ctx.client.query_df(f"SELECT id, text FROM {ctx.namespace}.notam_test ORDER BY id")
    clf = make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2)), LinearSVC())
    clf.fit(tr["text"].astype(str), tr["category"].astype(str))
    preds = clf.predict(te["text"].astype(str))
    out = pd.DataFrame({"id": te["id"].astype("uint32"), "predicted": preds})
    ctx.client.command(
        f"CREATE TABLE {ctx.namespace}.notam_pred (id UInt32, predicted String) "
        f"ENGINE = MergeTree ORDER BY id"
    )
    ctx.client.insert_df(f"{ctx.namespace}.notam_pred", out)
    return None


PROMPT = """aviation.notam holds NOTAM texts labeled with one of 13 categories, in a 'train' and a
'test' split. Your scratch database has a table `notam_test(id UInt64, text String)` — the test
NOTAMs WITHOUT labels.

Train a text classifier on aviation.notam WHERE split = 'train' (feature: `text`; target:
`category`), predict the category for every row of notam_test, and write your predictions to a table
named exactly `notam_pred(id UInt32, predicted String)` in your scratch database — one row per test
id, `predicted` being one of the 13 category names. You must beat the majority-class baseline by a
clear margin. Use run_python (pandas + scikit-learn are available). Then call finish.
"""

PROBLEM = AgentProblem(
    id="ds_notam_classify", category="ds", difficulty="hard",
    title="NOTAM text classification", prompt=PROMPT,
    check=check, reference=reference, setup=setup, max_steps=24,
    tags=("ml", "text", "notam"),
)
