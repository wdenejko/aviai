"""eval/v1 split-assignment tests — the three buckets are disjoint and correct.

Only the pure `assign_split` is exercised here; the full freeze needs the 152 MB corpus
and runs as a one-off, not in CI. Getting this function right is what guarantees the eval
set never overlaps the future training set (station-holdout AND time-holdout)."""

from avtext.harness.freeze import assign_split

_HELD = frozenset({"ENVA", "KEKM"})
_BOUNDARY = "2026-05-01"


def test_held_out_station_is_unseen_station_regardless_of_time():
    assert assign_split("ENVA", "2023-08-01 00:00:00", _HELD, _BOUNDARY) == "unseen_station"
    assert assign_split("ENVA", "2026-07-28 23:00:00", _HELD, _BOUNDARY) == "unseen_station"


def test_training_station_after_boundary_is_unseen_time():
    assert assign_split("EPGD", "2026-05-01 00:00:00", _HELD, _BOUNDARY) == "unseen_time"
    assert assign_split("EPGD", "2026-06-15 12:00:00", _HELD, _BOUNDARY) == "unseen_time"


def test_training_station_before_boundary_is_train():
    assert assign_split("EPGD", "2026-04-30 23:59:00", _HELD, _BOUNDARY) == "train"
    assert assign_split("EPGD", "2023-08-01 00:00:00", _HELD, _BOUNDARY) == "train"
