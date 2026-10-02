"""Tests for the replay and code checks (ADR-004 Revision 2, action item 6). The sandbox is a stub
here: what is ours is which reply reaches it, with what program, and what its result decides."""

from __future__ import annotations

import pytest
from dsbench.sftgen import replay_verify as rv


@pytest.mark.parametrize("answer, number", [
    ("40% of 300 is 120.\n\n**Answer:** \\boxed{120}", "120"),
    ("So she pays \\boxed{\\$1,250.50} in total.", "1250.50"),
    ("\\boxed{18 \\text{ dollars}}, from 3 + 15", "18"),  # the boxed one, not the last written
    ("The temperature falls to -5 degrees.", "-5"),
    ("He has 12 apples and gives away 4, so 8 remain.", "8"),
    ("\\boxed{x}, where x is 7", "7"),  # a box with no number: the last number written
    ("No number at all.", None),
])
def test_the_final_number_is_the_boxed_one_else_the_last_written(answer, number):
    assert rv.final_number(answer) == number


def test_numbers_compare_as_numbers():
    assert rv.same_number("120.0", "120") and not rv.same_number("121", "120")
    assert not rv.same_number(None, "120")


def test_the_program_is_the_defining_block_without_its_demo_then_every_test():
    item = {"verify": {"entry_point": "double", "tests": ["assert double(2) == 4",
                                                          "assert double(0) == 0"]}}
    answer = ("Here it is:\n```python\ndef double(x):\n    return 2 * x\n\nprint(double(3))\n```\n"
              "And a usage example:\n```python\nprint(double(5))\n```")
    program = rv.program(item, answer)
    assert "def double(x):" in program and "print(" not in program
    assert program.endswith("assert double(2) == 4\nassert double(0) == 0\n")


def _items():
    def item(pool, i, content="q", **verify):
        out = {"id": f"{pool}:{i}", "pool": pool,
               "messages": [{"role": "user", "content": content}]}
        if verify:
            out["verify"] = verify
        return out

    ner = ('Return {"Gene": [...]}. Only output the JSON object and do not include any '
           'additional text.')
    return [
        item("oasst1", 0), item("oasst1", 1), item("oasst1", 2), item("oasst1", 3),
        item("gsm8k", 0, gold_answer="120"), item("gsm8k", 1, gold_answer="120"),
        item("opencoder_edu", 0, entry_point="f", tests=["assert f() == 1"]),
        item("opencoder_edu", 1, entry_point="f", tests=["assert f() == 1"]),
        item("opencoder_edu", 2, entry_point="f", tests=["assert f() == 1"]),
        item("sciriff", 0, ner), item("sciriff", 1, ner), item("sciriff", 2, "Summarise it."),
        item("swe_swiss", 0),
    ]


def _gen(i, answer="an answer", reasoning="some thought", finish="stop", error=""):
    return {"id": i, "answer": answer, "reasoning": reasoning, "finish_reason": finish,
            "error": error}


def test_each_pool_keeps_the_replies_its_check_passes():
    gens = {r["id"]: r for r in [
        _gen("oasst1:0"),
        _gen("oasst1:1", finish="length"),  # cut by the block's budget
        _gen("oasst1:2", reasoning=""),
        _gen("oasst1:3", answer="ok </think> and again"),
        _gen("gsm8k:0", answer="So \\boxed{120}."), _gen("gsm8k:1", answer="So \\boxed{100}."),
        _gen("opencoder_edu:0", answer="```python\ndef f():\n    return 1\n```"),
        _gen("opencoder_edu:1", answer="```python\ndef f():\n    return 2\n```"),
        _gen("opencoder_edu:2", finish="length"),  # unfinished: never reaches the sandbox
        _gen("sciriff:0", answer=' {"Gene": ["BRCA1"]}\n'),
        _gen("sciriff:1", answer='```json\n{"Gene": ["BRCA1"]}\n```'),
        _gen("sciriff:2", answer="A summary, in prose."),  # no JSON asked for
        # swe_swiss:0 never answered: a run cut short checks what it answered
    ]}
    sent = []

    def sandbox(jobs):
        sent.extend(jobs)
        return {j["key"]: {"passed": "return 1" in j["program"],
                           "status": "ok" if "return 1" in j["program"] else "error"}
                for j in jobs}

    rows, report = rv.verify(_items(), gens, sandbox)
    checks = {r["id"]: r["check"] for r in rows}
    assert {i for i, c in checks.items() if c["ok"]} == {
        "oasst1:0", "gsm8k:0", "opencoder_edu:0", "sciriff:0", "sciriff:2"}
    assert checks["oasst1:1"]["why"] == "unfinished: length"
    assert checks["oasst1:2"]["why"] == "no reasoning"
    assert checks["oasst1:3"]["why"] == "think tag in the answer"
    assert checks["gsm8k:1"] == {"ok": False, "why": "final number 100, gold 120",
                                 "finished": True, "gold_match": False}
    assert checks["sciriff:1"]["why"] == "not the JSON alone"
    assert checks["opencoder_edu:1"]["why"] == "tests: error"
    assert [j["key"] for j in sent] == ["opencoder_edu:0", "opencoder_edu:1"]
    assert sent[0]["kind"] == "program" and sent[0]["timeout"] == rv.TEST_TIMEOUT_S
    assert "swe_swiss:0" not in checks
    assert report["pools"]["oasst1"] == {
        "items": 4, "answered": 4, "finished": 1, "kept": 1,
        "failed": {"unfinished: length": 1, "no reasoning": 1, "think tag in the answer": 1}}
    assert report["pools"]["gsm8k"]["failed"] == {"final number ...": 1}
    assert report["pools"]["swe_swiss"]["answered"] == 0
    assert (report["kept"], report["answered"], report["items"]) == (5, 12, 13)
