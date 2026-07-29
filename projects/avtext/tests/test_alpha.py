"""Krippendorff's alpha tests — a hand-computed value plus edge cases.

The 0.444 case is worked by hand in the comment so the implementation is pinned to
arithmetic, not to its own output."""

import pytest

from avtext.consensus import krippendorff_alpha


def test_perfect_agreement_is_one():
    assert krippendorff_alpha([[1, 1, 1], [2, 2, 2], ["a", "a"]]) == 1.0


def test_hand_computed_value():
    # units [[1,1],[2,2],[1,2]]: coincidences (1,1)=2 (2,2)=2 (1,2)=(2,1)=1;
    # marginals n1=n2=3, n=6; off-diag=2; alpha = 1 - 2*(6-1)/(36-18) = 1 - 10/18.
    assert krippendorff_alpha([[1, 1], [2, 2], [1, 2]]) == pytest.approx(0.4444, abs=1e-3)


def test_missing_ratings_are_ignored():
    # unit with a single rating is dropped; the rest all agree -> alpha 1.0
    assert krippendorff_alpha([[1, 1], [1, None], [1, 1]]) == 1.0


def test_insufficient_data_returns_none():
    assert krippendorff_alpha([[1], [2]]) is None  # no unit has >= 2 ratings
    assert krippendorff_alpha([]) is None
