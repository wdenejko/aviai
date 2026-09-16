"""DS expert: a deliberate data-leakage trap (ADR-002 backlog #9).

Trap hypothesis: the data has many ENTITIES, several rows each, and each entity's rows are a tight
cluster in feature space sharing one label. A random K-fold split puts rows of the same entity in
both train and test, so a high-capacity model (here KNN) just recognises the cluster and scores
~0.95 -- pure leakage. The honest question is "how well does this generalise to UNSEEN entities?",
answered by GroupKFold on `entity`, which scores far lower.

The checker asserts the score is in the HONEST band. So the obvious `cross_val_score(pipe, X, y,
cv=5)` (or a GroupKFold with the `groups=` argument forgotten -> sklearn raises) does NOT pass:
a high score is evidence of leakage, not skill. The band is a property check, kept wide enough to
be stable across sklearn versions but far below the leaky score. Calibrated on these fixtures:
honest GroupKFold ~0.65, leaky cv=5 ~0.92 (chance ~0.60) -- the honest score sits only just above
chance ON PURPOSE, because the whole point is that this model barely generalises to unseen
entities and only leakage makes it look good.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsbench.schema import Problem

_HONEST_LO, _HONEST_HI = 0.55, 0.85  # brackets honest ~0.65, excludes leaky ~0.89-0.92


def make_inputs() -> dict:
    rng = np.random.default_rng(17)
    n_entities, per_entity, row_noise = 30, 8, 0.10
    centers = rng.normal(0.0, 1.0, size=(n_entities, 2))
    # Label is a NOISY function of the entity centre -> cross-entity generalisation is weak, while
    # within-entity clusters are near-perfectly separable (that gap is exactly the leakage).
    logits = centers[:, 0] + centers[:, 1] + rng.normal(0.0, 1.0, size=n_entities)
    ent_label = (logits > 0).astype(int)
    frames = []
    for e in range(n_entities):
        pts = centers[e] + rng.normal(0.0, row_noise, size=(per_entity, 2))
        frames.append(pd.DataFrame({
            "entity": e,
            "f1": pts[:, 0],
            "f2": pts[:, 1],
            "label": ent_label[e],
        }))
    df = pd.concat(frames, ignore_index=True)
    return {"df": df.sample(frac=1.0, random_state=3).reset_index(drop=True)}


def check(result) -> tuple[bool, str]:
    try:
        s = float(result)
    except (TypeError, ValueError):
        return False, "did not return a float"
    ok = _HONEST_LO <= s <= _HONEST_HI
    return ok, "" if ok else (
        f"mean accuracy {s:.3f} outside honest band [{_HONEST_LO}, {_HONEST_HI}] -- "
        f"a leaky random-split (cv=5) scores ~0.92 and must fail; honest GroupKFold ~0.65"
    )


PROMPT = """You are given a pandas DataFrame `df` with columns: entity (int), f1 (float),
f2 (float), label (0/1). The label is a property of the ENTITY (all rows of an entity share it),
and each entity's rows are tightly clustered in (f1, f2). You must estimate how well a model
generalises to ENTITIES IT HAS NEVER SEEN -- so a plain random split would leak entity identity
and overstate accuracy.

Write `solve(df)` that:
  - builds Pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)) on features [f1, f2],
  - runs 5-fold GroupKFold cross-validation grouping by `entity` (pass the groups correctly),
    scoring accuracy,
  - returns the MEAN cross-validation accuracy as a float.

Return only a single ```python code block defining `solve`.
"""

REFERENCE = '''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
def solve(df):
    X = df[["f1", "f2"]]; y = df["label"]; groups = df["entity"]
    pipe = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))])
    scores = cross_val_score(pipe, X, y, cv=GroupKFold(n_splits=5), groups=groups)
    return float(scores.mean())
'''

PROBLEM = Problem(
    id="ds_expert_01", category="ds", difficulty="expert",
    title="Leakage trap: group-aware CV", prompt=PROMPT, mode="python",
    make_inputs=make_inputs, check=check, reference=REFERENCE,
    tags=("sklearn", "leakage", "groupkfold"),
)
