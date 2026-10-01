"""Tests for the reasoning prefill's box-side half (sftgen/prefill.py): where a plain trace is cut,
and the `recall` items built from it. The traces are shortened from the hint pilot's plain
ClickHouse replies (reports/gate-evals/20260930-target-a-hints-pilot.md).
"""
from __future__ import annotations

import pytest
from dsbench.sftgen import prefill

ROW = "A-weekday-numbering-clickhouse-gym_checkins-20760945"


def _cut_text(text: str) -> str | None:
    at = prefill.cut(text)
    return None if at is None else text[at:]


@pytest.mark.parametrize("trace, rest", [
    # Mid-paragraph: the cut takes the whole sentence that states the numbering.
    ("The user wants check-ins on Saturdays.\n\n1.  **Identify the table**: `gym_checkins`.\n"
     "2.  **Filter condition**: Saturdays. In ClickHouse, `dayOfWeek` function returns a number "
     "where Sunday=1, Monday=2, ..., Saturday=7.",
     "In ClickHouse, `dayOfWeek` function returns"),
    # A list item: the marker stays, and the convention becomes the item. The header's "day of
    # week" is prose, not the function.
    ("4.  **Determine the function to extract day of week**:\n    *   ClickHouse has "
     "`dayOfWeek(timestamp)`. It returns 1 for Sunday, 2 for Monday, ..., 7 for Saturday.",
     "ClickHouse has `dayOfWeek(timestamp)`."),
    # A numbering stated before any function is named.
    ("Plan the filter.\nSaturdays are day 7 in ClickHouse, since Sunday is 1.\nSo use "
     "toDayOfWeek.",
     "Saturdays are day 7 in ClickHouse"),
    # A query drafted in a code block: cut before its fence, not inside the SQL.
    ("Draft:\n```sql\nSELECT count() FROM t WHERE toDayOfWeek(ts) = 1\n```\nDone.",
     "```sql\nSELECT count()"),
    # The question's own "Sundays" is not a numbering.
    ("The user asks: On Sundays specifically, how many orders were there in total? I'll use "
     "toDayOfWeek(order_ts) = 7.",
     "I'll use toDayOfWeek(order_ts) = 7."),
    ("toDayOfWeek returns 1 for Sunday.", "toDayOfWeek returns 1 for Sunday."),
    ("Count the rows that fall on a weekend, with a date function.", None),
])
def test_the_cut_is_the_sentence_that_first_turns_to_the_weekday(trace, rest):
    got = _cut_text(trace)
    assert got == rest if rest is None else got.startswith(rest)


def _plain(sample: int = 2) -> dict:
    return {"id": f"targetA:{ROW}#{sample}", "pool": "targetA",
            "messages": [{"role": "system", "content": "ClickHouse."},
                         {"role": "user", "content": "On Saturdays specifically, how many?"}],
            "prefill": "", "verify": {"row_id": ROW, "dialect": "clickhouse", "truth": 3},
            "meta": {"placement": "plain", "sample": sample, "recall": "RECALL."}}


def test_a_plain_reply_becomes_a_recall_item():
    reasoning = "Plan.\n- In ClickHouse, `toDayOfWeek` returns 1 for Sunday.\n- So = 7."
    item = prefill.splice(_plain(), {"id": "x", "error": "", "reasoning": reasoning})
    assert item["id"] == f"targetA_recall:{ROW}#2" and item["pool"] == "targetA_recall"
    assert item["prefill"] == "Plan.\n- RECALL."  # the base's opening, then the convention
    assert item["messages"] == _plain()["messages"]  # the prompt stays plain
    assert item["verify"] == {**_plain()["verify"], "recall": "RECALL."}
    assert item["meta"]["placement"] == "recall" and item["meta"]["cut_chars"] == len("Plan.\n- ")
    assert item["meta"]["spliced_from"] == f"targetA:{ROW}#2"


def test_no_recall_item_from_a_failed_reply_or_one_that_never_gets_there():
    assert prefill.splice(_plain(), {"id": "x", "error": "ReadTimeout", "reasoning": ""}) is None
    assert prefill.splice(_plain(), {"id": "x", "error": "", "reasoning": "Count rows."}) is None


def test_splice_all_takes_the_retried_reply_and_counts_the_rest():
    start = {**_plain(1), "id": f"targetA_start:{ROW}#1", "meta": {"placement": "start"}}
    items = [_plain(1), start, _plain(2)]
    records = [
        {"id": f"targetA:{ROW}#1", "error": "ReadTimeout"},
        {"id": f"targetA:{ROW}#1", "error": "", "reasoning": "Plan. toDayOfWeek is 1 for Sunday."},
        {"id": f"targetA:{ROW}#2", "error": "", "reasoning": "Count the rows."},
    ]
    out, counts = prefill.splice_all(items, records)
    assert [i["id"] for i in out] == [f"targetA_recall:{ROW}#1"]
    assert out[0]["prefill"] == "Plan. RECALL."
    assert counts == {"plain": 2, "spliced": 1, "no_cut": 1}
