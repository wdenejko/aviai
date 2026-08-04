"""NOTAM schema tests — the no-Chinese guard, area_type exclusion, and the category contract."""

import pytest
from pydantic import ValidationError

from avtext.schema.notam import (
    NOTAM_FIELDS,
    ROW_CATEGORIES,
    NotamExtraction,
    has_chinese,
    normalize_row,
)


def test_has_chinese_detects_anywhere():
    assert has_chinese("本场不接收")  # raw string
    assert has_chinese({"area_type": "限制区", "atc": "OK"})  # value in a dict
    assert has_chinese([{"x": None}, {"y": "禁"}])  # nested
    assert not has_chinese("RWY 13/31 CLSD")
    assert not has_chinese({"airport": "MMRX", "runway": "13", "tora": None})


def test_normalize_row_drops_area_type_and_remaps_keys():
    # area_type is 100%-Chinese and NOT in the schema -> dropped; other fields kept + stripped.
    row = normalize_row(
        "area", {"area_type": "限制区", "area_summary": " restricted ", "atc": "", "fpl": "Y"}
    )
    assert "area_type" not in row  # excluded from NOTAM_FIELDS["area"]
    assert row["area_summary"] == "restricted"  # stripped
    assert row["atc"] is None  # "" -> None
    assert row["fpl"] == "Y"
    assert set(row) == set(NOTAM_FIELDS["area"])  # exactly the schema fields
    assert not has_chinese(row)  # the Chinese field is gone


def test_normalize_row_remaps_capitalised_keys():
    # Knots emits 'Chart' / 'restriction_Type' etc.; we canonicalise to snake_case.
    r = normalize_row("procedure", {"Chart": "IAC", "airport": "KJFK"})
    assert r["chart"] == "IAC" and r["airport"] == "KJFK"


def test_schema_rejects_non_category_fields():
    NotamExtraction(
        id="X1/24",
        category="taxiway",
        raw_text="A) K E)TWY A CLSD",
        rows=[{"airport": "K", "taxiway": "A", "status_type": "clsd"}],
    )  # ok
    with pytest.raises(ValidationError):
        NotamExtraction(
            id="X2/24", category="taxiway", raw_text="...", rows=[{"runway": "13"}]
        )  # 'runway' is not a taxiway field


def test_flat_vs_row_categories():
    assert "runway" in ROW_CATEGORIES and "area" in ROW_CATEGORIES
    assert "airport" not in ROW_CATEGORIES  # flat (single extraction)
    assert set(NOTAM_FIELDS) == {
        "airport",
        "airway",
        "area",
        "light",
        "navigation",
        "procedure",
        "runway",
        "stand",
        "taxiway",
    }  # all nine categories present
