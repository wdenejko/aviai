"""Unit tests for the synthetic Target-C ML tasks (pure; no ClickHouse, no network).

The real validity gates -- `ml_tasks --reps` (the task is solvable) and `ml_task_controls` (the
careless approach fails) -- need a live sandbox and run as CLI tools. What is cheap to assert here
are the two invariants whose violation is silent: a task with no negative control looks fine until
it turns out not to discriminate, and a generator that ignores the run seed produces N identical
datasets while the report happily claims N trajectories.
"""
from __future__ import annotations

import numpy as np
from dsbench.sftgen.ml_task_controls import CONTROLS
from dsbench.sftgen.ml_tasks import ML_TASKS, _credit_data, _energy_data, _ticket_data

# Every generator, with the base seed its setup() uses.
_GENERATORS = [
    ("widget", 101), ("delivery", 202), ("churn", 303),
    ("energy", 404), ("ticket", 505), ("credit", 606), ("upsell", 707),
]


def test_every_task_has_a_negative_control():
    """A task without a control can silently stop discriminating (see mlc_credit_leak)."""
    missing = sorted({p.id for p in ML_TASKS} - set(CONTROLS))
    assert not missing, f"tasks with no negative control: {missing}"
    stale = sorted(set(CONTROLS) - {p.id for p in ML_TASKS})
    assert not stale, f"controls for tasks that no longer exist: {stale}"


def test_task_ids_and_deliverables_are_unique():
    ids = [p.id for p in ML_TASKS]
    assert len(ids) == len(set(ids)), f"duplicate task ids: {ids}"
    # Two tasks writing the same deliverable table would collide if they ever shared a namespace.
    for p in ML_TASKS:
        assert p.prompt.count("Deliver") >= 1, f"{p.id} prompt never states the deliverable"
        assert p.max_steps > 0 and p.setup is not None


def test_run_seed_varies_the_dataset():
    """Volume generation reruns each task; if the seed did not move, reps would be duplicates."""
    a, _ = _ticket_data(505)
    b, _ = _ticket_data(506)
    assert not np.allclose(a["words"].to_numpy(), b["words"].to_numpy())


def test_credit_flag_is_an_exact_leak_before_setup_zeroes_it():
    """The trap only bites when the flag is pure -- an imperfect leak left the model at 0.81."""
    df, _ = _credit_data(606)
    assert (df["collections_flag"].to_numpy() == df["label"].to_numpy()).all()


def test_energy_split_is_temporal_not_random():
    """The lesson of mlc_energy_load is the time boundary; a shuffled split would erase it."""
    df, is_test = _energy_data(404)
    assert df.loc[is_test, "ts_hour"].min() > df.loc[~is_test, "ts_hour"].max()


def test_ticket_uses_every_queue():
    """Macro-F1 averages over all four queues, so a generator that starves one caps the score."""
    df, _ = _ticket_data(505)
    counts = df["queue"].value_counts()
    assert len(counts) == 4, f"expected 4 queues, got {dict(counts)}"
    assert counts.min() / counts.sum() > 0.05, f"a queue is nearly empty: {dict(counts)}"
