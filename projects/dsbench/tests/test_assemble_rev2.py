"""Tests for Revision 2's assembler (ADR-004 Revision 2, action item 9): the records it makes from
checked replies, what it drops, and how it fills the budgets. The records go through the real
chat template (`test_tokenize_masked.TemplateTokenizer`), as the box's build will put them."""

from __future__ import annotations

import json
import random
from collections import Counter

import pytest
from dsbench.sftgen import assemble_rev2 as ar
from dsbench.sftgen import tokenize_masked as tm
from dsbench.sftgen.decontaminate import DenyList
from test_tokenize_masked import TOOLS, TemplateTokenizer


def _item(pool="oasst1", i=0, **extra):
    return {"id": f"{pool}:{i}", "pool": pool, "messages": [{"role": "user", "content": "Hi?"}],
            "meta": {"licence": "Apache-2.0", "redistributable": True}, **extra}


def _rec(item, reasoning="Think it over.", answer="Hello.", prompt=100, reply=50, ok=True,
         **extra):
    return {"id": item["id"], "pool": item["pool"], "error": "", "reasoning": reasoning,
            "answer": answer, "finish_reason": "stop", "prompt_tokens": prompt,
            "completion_tokens": reply, "sampling": {"temperature": 0.6, "seed": 7},
            "check": {"ok": ok, "why": "" if ok else "wrong", "finished": True}, **extra}


def _labelled(tok, record):
    ids, labels = tm.thinking_record(tok, record)
    return tok.text([i for i, lab in zip(ids, labels, strict=True) if lab != tm.IGNORE])


# --- records -----------------------------------------------------------------------------------


def test_a_kept_reply_becomes_a_record_the_template_labels():
    item = _item()
    row = ar.single_turn(item, _rec(item))
    assert row.tokens == 151 and row.reply == 50  # 100 + 50 and the newline after <|im_end|>
    assert row.record["messages"][-1] == {"role": "assistant", "content": "Hello.",
                                          "reasoning_content": "Think it over."}
    assert row.record["meta"]["answer"]["sampling"] == {"temperature": 0.6, "seed": 7}
    assert row.record["meta"]["source"]["licence"] == "Apache-2.0"
    tok = TemplateTokenizer()
    assert _labelled(tok, row.record) == "Think it over.\n</think>\n\nHello.<|im_end|>"


def test_a_tool_call_reply_keeps_its_tools_and_its_call():
    item = _item("tool_fit", tools=TOOLS)
    call = {"id": "c1", "type": "function",
            "function": {"name": "get_time", "arguments": '{"city": "Oslo"}'}}
    row = ar.single_turn(item, _rec(item, answer="", tool_calls=[call]))
    assert row.record["tools"] == TOOLS
    labelled = _labelled(TemplateTokenizer(), row.record)
    assert labelled.startswith("Think it over.\n</think>\n\n<tool_call>")
    assert "Oslo" in labelled and labelled.endswith("</tool_call><|im_end|>")


def test_target_a_trains_on_its_training_prompt_and_its_whole_trace():
    item = _item("targetA_recall", messages=[{"role": "user", "content": "Sundays?"}])
    rec = _rec(item, reasoning="Weekdays first. In ClickHouse, Sunday is 7. So = 7.",
               answer="```sql\nSELECT 1\n```", train_messages=[{"role": "user", "content": "Q"}])
    rec["check"] = {"status": "verified", "kept": True}
    row = ar.single_turn(item, rec)
    assert row.record["messages"][0] == {"role": "user", "content": "Q"}
    assert ar.passed(rec) and ar.pool_of("targetA_recall") == "targetA"


def test_the_check_decides_and_gsm8k_follows_decision_5():
    gold_miss = {"pool": "gsm8k", "check": {"ok": False, "finished": True, "gold_match": False}}
    assert not ar.passed(gold_miss) and ar.passed(gold_miss, gsm8k="finished")
    cut = {"pool": "gsm8k", "check": {"ok": False, "finished": False}}
    assert not ar.passed(cut, gsm8k="finished")
    assert not ar.passed({"pool": "oasst1", "check": {"ok": False, "finished": True}},
                         gsm8k="finished")  # the policy is GSM8K's alone


def test_target_a_s_doubting_traces_follow_decision_2():
    doubting = {"pool": "targetA_recall", "check": {"status": "verified", "kept": True,
                                                    "doubts": 2}}
    clean = {"pool": "targetA_recall", "check": {"status": "verified", "kept": True,
                                                 "doubts": 0}}
    assert not ar.passed(doubting) and ar.passed(doubting, target_a_doubts="keep")
    assert ar.passed(clean)
    plain_cell = {"pool": "targetA", "check": {"status": "verified", "kept": True}}
    assert ar.passed(plain_cell)  # no doubts are counted where Sunday = 1 isn't the wrong belief


def test_a_target_a_count_that_matched_by_chance_is_dropped():
    # target_a_eval recheck: right on its own table, wrong on two more of its domain
    lucky = {"pool": "targetA_recall", "check": {"status": "verified", "kept": True, "doubts": 0,
                                                 "recheck": {"held": False}}}
    held = {"pool": "targetA_recall", "check": {"status": "verified", "kept": True, "doubts": 0,
                                                "recheck": {"held": True}}}
    assert not ar.passed(lucky) and not ar.passed(lucky, target_a_doubts="keep")
    assert ar.passed(held)


def _loop():
    call = {"type": "function", "function": {"name": "get_time", "arguments": '{"city": "Oslo"}'}}
    messages = [
        {"role": "user", "content": "What time is it in Oslo?"},
        {"role": "assistant", "content": "", "reasoning_content": "Ask the tool.",
         "tool_calls": [call, call]},
        {"role": "tool", "content": "12:00"},
        {"role": "tool", "content": "SQL error: Code: 46. Unknown function stddev"},
        {"role": "assistant", "content": "", "reasoning_content": "Again, one call.",
         "tool_calls": [call]},
        {"role": "tool", "content": "12:00"},
        {"role": "assistant", "content": "It is noon.", "reasoning_content": "Done."},
    ]
    return {"tools": TOOLS, "messages": messages,
            "meta": {"id": "C-x-1", "selection": {"served_tokens": 3728},
                     "turns": [{"completion_tokens": 200}, {"completion_tokens": 122},
                               {"completion_tokens": 30}]}}


def test_a_target_c_loop_is_counted_as_its_selection_counted_it():
    row = ar.trajectory(_loop(), mask_failed=False)
    assert (row.tokens, row.reply, row.record["meta"]["pool"]) == (3729, 352, "target_c")
    assert row.record["meta"]["masked_turns"] == []


def test_a_failed_call_is_seen_in_any_tool_message_that_answers_its_turn():
    from dsbench.agentic.tools import is_error

    assert ar.failed_turns(_loop()["messages"]) == [1]  # one of its two parallel calls failed
    assert is_error("Traceback (most recent call last):\n  ...\nKeyError: 'x'")
    assert is_error('File "<stdin>", line 31\n    x = [\'a\', b\']\nSyntaxError: ...')
    assert is_error("run_python error: timed out (90s)") and is_error("tool-call error: bad JSON")
    assert not is_error("caught it: Error: the column is empty")  # the script ran
    assert not is_error("(0 rows)")


def test_decision_7_keeps_a_failed_turn_as_context_without_training_it():
    row = ar.trajectory(_loop())  # masked by default, as proposed
    assert row.record["meta"]["masked_turns"] == [1] and row.reply == 152  # 122 + 30
    assert row.record["messages"][1]["loss"] is False
    tok = TemplateTokenizer()
    labelled = _labelled(tok, row.record)
    assert "Ask the tool." not in labelled  # the failed turn is context only
    assert "Again, one call." in labelled and labelled.endswith("It is noon.<|im_end|>")
    # the row's text is the same either way: only the labels differ
    plain_ids, _ = tm.thinking_record(tok, ar.trajectory(_loop(), mask_failed=False).record)
    assert tm.thinking_record(tok, row.record)[0] == plain_ids


def test_a_row_whose_every_turn_is_masked_is_rejected():
    record = ar.trajectory(_loop()).record
    record["messages"] = [m if m["role"] != "assistant" else {**m, "loss": False}
                          for m in record["messages"]]
    with pytest.raises(tm.RowRejected, match="no_labelled_turn"):
        tm.thinking_record(TemplateTokenizer(), record)


def test_load_joins_each_reply_to_its_item(tmp_path):
    items = [_item(i=i) for i in range(3)]
    (tmp_path / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    recs = [_rec(items[0]), _rec(items[1], ok=False), _rec(items[2])]
    (tmp_path / "v.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    rows, read = ar.load_verified(tmp_path / "items.jsonl", tmp_path / "v.jsonl", "gold")
    assert [r.record["meta"]["id"] for r in rows] == ["oasst1:0", "oasst1:2"]
    assert read == Counter({"oasst1": 3})
    (tmp_path / "v.jsonl").write_text(json.dumps(_rec(_item(i=9))) + "\n")
    with pytest.raises(SystemExit, match="is not in"):
        ar.load_verified(tmp_path / "items.jsonl", tmp_path / "v.jsonl", "gold")


# --- budgets -----------------------------------------------------------------------------------


def test_budgets_split_each_line_by_its_shares():
    by = {p.name: ar.budget(p, ar.POOLS) for p in ar.POOLS}
    assert by["opencoder_edu"] == 1_875_000 and by["swe_swiss"] == 625_000
    assert (by["oasst1"], by["gsm8k"], by["flan_v2"]) == (1_600_000, 800_000, 0)
    assert sum(by.values()) == 10_000_000
    assert ar.budget(ar.POOLS[0], ar.POOLS, scale=1.5) == 1_200_000
    wider = tuple(p if p.name != "flan_v2" else ar.Pool("flan_v2", "replay", 0.25)
                  for p in ar.POOLS)  # a share given to FLAN v2 comes out of the others
    assert ar.budget(next(p for p in wider if p.name == "oasst1"), wider) == 1_280_000


def test_a_line_can_be_opened_for_one_assembly_and_the_others_stay():
    # Revision 2.1 (ADR-004, the recall round): Target A's line opened, every other line as it was.
    lines = ar.bucket_tokens(["target_a=1200000"])
    assert lines["target_a"] == 1_200_000 and ar.BUCKET_TOKENS["target_a"] == 800_000
    assert {k: v for k, v in lines.items() if k != "target_a"} == {
        k: v for k, v in ar.BUCKET_TOKENS.items() if k != "target_a"}
    by = {p.name: ar.budget(p, ar.POOLS, buckets=lines) for p in ar.POOLS}
    assert by["targetA"] == 1_200_000 and by["oasst1"] == 1_600_000
    assert ar.bucket_tokens([]) == ar.BUCKET_TOKENS  # the defaults rebuild Revision 2
    with pytest.raises(SystemExit, match="unknown line"):
        ar.bucket_tokens(["targetA=1"])  # a pool's name, not its line's
    with pytest.raises(SystemExit, match="whole number"):
        ar.bucket_tokens(["target_a=1.2M"])


POOLS = (ar.Pool("oasst1", "replay", 0.5), ar.Pool("gsm8k", "replay", 0.5),
         ar.Pool("target_c", "target_c"))


def test_assembly_drops_what_cannot_train_and_fills_each_budget(monkeypatch):
    monkeypatch.setattr(ar, "BUCKET_TOKENS", {"replay": 2000, "target_c": 5000})
    rows = [ar.single_turn(_item(i=i), _rec(_item(i=i), prompt=100, reply=99)) for i in range(15)]
    rows.append(ar.single_turn(_item(i=20), _rec(_item(i=20), prompt=4000, reply=4191)))  # 8192
    rows.append(ar.single_turn(_item(i=21), _rec(_item(i=21), prompt=None)))  # no count
    rows.append(ar.single_turn(_item(i=22), _rec(_item(i=22), reasoning="So 44.1 it is.")))
    rows += [ar.single_turn(_item("gsm8k", i), _rec(_item("gsm8k", i), prompt=100, reply=99))
             for i in range(3)]
    deny = DenyList(grams=set(), identifiers=(), answers=("44.1",))
    mixture, result = ar.assemble(rows, Counter(oasst1=18, gsm8k=3), deny, pools=POOLS, seed=1)
    by = {e["pool"]: e for e in result["pools"]}
    oasst = by["oasst1"]
    assert (oasst["budget"], oasst["kept_by_check"], oasst["over_block"], oasst["uncounted"]) == (
        1000, 18, 1, 1)
    assert oasst["contaminated"] == {"numeric-answer": 1}
    assert (oasst["eligible"], oasst["rows"], oasst["tokens"]) == (15, 5, 1000)  # 5 x 200
    assert oasst["left_over_rows"] == 10 and oasst["shortfall"] == 0
    gsm = by["gsm8k"]
    assert (gsm["rows"], gsm["tokens"], gsm["shortfall"]) == (3, 600, 400)  # all it has
    assert by["target_c"]["shortfall"] == 5000  # no input: the whole budget, never padded
    assert result["total"] == {"rows": 8, "tokens": 1600, "reply_tokens": 8 * 99,
                               "budget": 7000}
    assert result["buckets"]["replay"]["pct_of_mixture"] == 100.0
    assert {r["meta"]["mix_pool"] for r in mixture} == {"oasst1", "gsm8k"}
    # each record carries the counts it was selected by, for build_masked_dataset to check
    assert {(r["meta"]["mix_tokens"], r["meta"]["mix_reply"]) for r in mixture} == {(200, 99)}
    again, _ = ar.assemble(rows, Counter(), deny, pools=POOLS, seed=1)
    assert [r["meta"]["id"] for r in again] == [r["meta"]["id"] for r in mixture]

    # A line opened for one assembly changes only its own pools' rows: every pool draws from its
    # own seeded generator, so a pool still bound by its line picks the same rows.
    split = (ar.Pool("oasst1", "replay"), ar.Pool("gsm8k", "math"))
    more = rows + [ar.single_turn(_item("gsm8k", i), _rec(_item("gsm8k", i), prompt=100, reply=99))
                   for i in range(3, 12)]
    before, _ = ar.assemble(more, Counter(), deny, pools=split, seed=1,
                            buckets={"replay": 1000, "math": 1000})
    after, opened = ar.assemble(more, Counter(), deny, pools=split, seed=1,
                                buckets={"replay": 10_000, "math": 1000})
    by_line = {e["pool"]: e for e in opened["pools"]}
    assert (by_line["oasst1"]["rows"], by_line["oasst1"]["left_over_rows"]) == (15, 0)
    assert (by_line["gsm8k"]["rows"], by_line["gsm8k"]["left_over_rows"]) == (5, 7)
    assert ({r["meta"]["id"] for r in after if r["meta"]["mix_pool"] == "gsm8k"}
            == {r["meta"]["id"] for r in before if r["meta"]["mix_pool"] == "gsm8k"})


def test_rows_from_a_pool_with_no_budget_stop_the_assembly():
    rows = [ar.single_turn(_item("mystery"), _rec(_item("mystery")))]
    with pytest.raises(SystemExit, match="no budget"):
        ar.assemble(rows, Counter(), None, pools=POOLS)


def test_selection_stops_once_the_budget_is_reached():
    rows = [ar.Row({"meta": {}}, 300, 0) for _ in range(10)]
    assert len(ar.select(rows, 1000, random.Random(0))) == 4  # 900 < 1000, then 1200
    assert ar.select(rows, 0, random.Random(0)) == []
