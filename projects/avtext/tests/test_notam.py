"""NOTAM schema tests — the no-Chinese guard, area_type exclusion, and the category contract."""

import pytest
from pydantic import ValidationError

from avtext.schema.notam import (
    NOTAM_CLASSES,
    NOTAM_FIELDS,
    ROW_CATEGORIES,
    NotamClassification,
    NotamExtraction,
    has_chinese,
    normalize_row,
)


def test_notam_classification_schema():
    # DEEL-AI classification: 13 top-level classes, one label per NOTAM.
    assert len(NOTAM_CLASSES) == 13 and "Runway" in NOTAM_CLASSES and "Wildlife" in NOTAM_CLASSES
    c = NotamClassification(id="cls-1", raw_text="ILS IJB RWY 15 U/S", label="Landing_Navaids")
    assert c.label == "Landing_Navaids"


def test_has_chinese_detects_anywhere():
    assert has_chinese("本场不接收")  # raw string
    assert has_chinese({"area_type": "限制区", "atc": "OK"})  # value in a dict
    assert has_chinese([{"x": None}, {"y": "禁"}])  # nested
    assert not has_chinese("RWY 13/31 CLSD")
    assert not has_chinese({"airport": "MMRX", "runway": "13", "tora": None})


def test_normalize_row_remaps_area_type_enum():
    # area_type is a 6-value Chinese enum -> remapped to English + kept (recovers area, no Chinese).
    row = normalize_row(
        "area", {"area_type": "多边形", "area_summary": " restricted ", "atc": "", "fpl": "Y"}
    )
    assert row["area_type"] == "polygon"  # 多边形 -> polygon (remapped, kept)
    assert row["area_summary"] == "restricted"  # stripped
    assert row["atc"] is None  # "" -> None
    assert row["fpl"] == "Y"
    assert set(row) == set(NOTAM_FIELDS["area"])  # exactly the schema fields (incl. area_type)
    assert not has_chinese(row)  # remapped -> no Chinese


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
        "rvr",
        "standard",
        "stand",
        "taxiway",
    }  # all eleven categories present (OpenNOTAM adds rvr + standard)


def _rw(**kw):
    """A runway row with all schema fields (missing -> None)."""
    return {f: kw.get(f) for f in NOTAM_FIELDS["runway"]}


def test_score_notam_perfect_and_order_invariant():
    from avtext.harness.score import Outcome
    from avtext.harness.score_notam import row_count_match, score_notam_record

    ref = {
        "category": "runway",
        "rows": [
            _rw(airport="MMRX", runway="13", status_type="clsd"),
            _rw(airport="MMRX", runway="31", status_type="clsd"),
        ],
    }
    # prediction identical but ROWS SWAPPED -> greedy match should still score perfect
    pred = {"rows": [ref["rows"][1], ref["rows"][0]]}
    s = score_notam_record(ref, pred, id="MMRX_1/24")
    assert s.exact_match
    assert row_count_match(ref, pred)
    assert not any(o in (Outcome.WRONG, Outcome.HALLUCINATE) for o in s.outcomes.values())


def test_score_notam_extra_row_hallucinates_missing_abstains():
    from avtext.harness.score import Outcome
    from avtext.harness.score_notam import score_notam_record

    ref = {"category": "runway", "rows": [_rw(airport="K", runway="9", status_type="clsd")]}
    invents = {"rows": [ref["rows"][0], _rw(airport="K", runway="27", status_type="clsd")]}
    assert any(o is Outcome.HALLUCINATE for o in score_notam_record(ref, invents).outcomes.values())
    drops = {"rows": []}
    assert any(o is Outcome.ABSTAIN for o in score_notam_record(ref, drops).outcomes.values())


def test_notam_prompt_roundtrip_is_perfect():
    # reference -> target JSON -> parse -> score must be perfect (prompt/parser/scorer consistency)
    import json

    from avtext.harness.prompt_notam import parse_prediction_notam
    from avtext.harness.score import Outcome
    from avtext.harness.score_notam import score_notam_record

    ref = {
        "category": "taxiway",
        "rows": [
            {
                f: v
                for f, v in zip(
                    NOTAM_FIELDS["taxiway"], ["EDDF", "A", "clsd", None, "B"], strict=True
                )
            }
        ],
    }
    target = json.dumps({"rows": ref["rows"]})
    pred = parse_prediction_notam("JSON:\n" + target, "taxiway")
    s = score_notam_record(ref, pred)
    assert s.exact_match
    assert not any(o in (Outcome.WRONG, Outcome.HALLUCINATE) for o in s.outcomes.values())


def _replayer(outputs):
    """A fake `complete` that returns the queued model outputs in call order (concurrency=1)."""
    it = iter(outputs)

    def complete(_prompt):
        return next(it)

    return complete


def test_runner_extraction_perfect_model_scores_clean():
    # A model that emits the gold rows verbatim must score EM=all, 0 invalid, full row-count match.
    import json

    from avtext.harness.runner_notam import run_extraction

    records = [
        {
            "id": "rw-1",
            "category": "runway",
            "raw": "RWY 13/31 CLSD",
            "reference": {
                "category": "runway",
                "rows": [_rw(airport="KJFK", runway="13", status_type="clsd")],
            },
        }
    ]
    golds = [json.dumps({"rows": r["reference"]["rows"]}) for r in records]
    scores, preds, extra = run_extraction(records, _replayer(golds), 1)
    assert extra["n_invalid"] == 0
    assert extra["n_row_match"] == len(records)
    assert all(s.exact_match for s in scores)


def test_runner_classification_accuracy_and_invalid():
    # Perfect classes -> 100% accuracy; empty output -> counted invalid, not a crash.
    from avtext.harness.runner_notam import run_classification

    records = [
        {"id": "c1", "raw": "ILS RWY 15 U/S", "reference": "Landing_Navaids"},
        {"id": "c2", "raw": "TWY A CLSD", "reference": "Taxiway"},
    ]
    good = _replayer([r["reference"] for r in records])
    metrics, _ = run_classification(records, good, 1)
    assert metrics["accuracy"] == 1.0 and metrics["invalid"] == 0

    junk = _replayer(["", "not-a-class-xyz"])
    m2, _ = run_classification(records, junk, 1)
    assert m2["accuracy"] == 0.0 and m2["invalid"] >= 1
