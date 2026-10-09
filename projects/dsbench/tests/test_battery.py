"""Tests for the acceptance battery's own glue: extraction, statistics, item building, requests.

The reference scorers (IFEval, LiveCodeBench, BFCL, DS-1000, HumanEval+ tests) are upstream code
and are validated on the box by scoring their own gold solutions (`score.py --gold`). What is ours,
and therefore tested here, is everything around them.
"""

from __future__ import annotations

import base64
import json
import pickle
import zlib

import pytest
from dsbench.battery import contamination, extract, prepare, stats
from dsbench.battery.generate import request_body
from dsbench.battery.items import Item, load_items, save_items

# --- extraction --------------------------------------------------------------------------------


def test_humaneval_program_strips_demo_code_and_keeps_prompt_imports():
    prompt = "from typing import List\n\ndef add(xs: List[int]) -> int:\n    \"\"\"Sum.\"\"\"\n"
    reply = ("Here you go:\n```python\ndef add(xs: List[int]) -> int:\n    return sum(xs)\n\n"
             "if __name__ == '__main__':\n    print(add([1, 2]))\nassert add([]) == 0\n```")
    program = extract.humaneval_program(reply, prompt, "add")
    assert program.startswith("from typing import List\n")
    assert "return sum(xs)" in program
    assert "__main__" not in program and "assert" not in program and "print" not in program


def test_humaneval_program_body_only_reply_falls_back_to_prompt_plus_body():
    prompt = "def inc(x):\n    \"\"\"Add one.\"\"\"\n"
    program = extract.humaneval_program("```python\n    return x + 1\n```", prompt, "inc")
    ns: dict = {}
    exec(program, ns)  # noqa: S102 - our own fixed test string
    assert ns["inc"](1) == 2


def test_ds1000_postprocess_matches_upstream_rules():
    assert extract.ds1000_code("<code>\nresult = df.iloc[List]\n</code>") == \
        "\nresult = df.iloc[List]\n"
    assert extract.ds1000_code("```python\nresult = 1\n```\ntrailing") == "\nresult = 1\n"
    assert extract.ds1000_code("result = 2\nEND SOLUTION\nx") == "result = 2"


def test_lcb_code_takes_the_last_fenced_block():
    reply = "```python\nfirst\n```\ntext\n```python\nsecond()\n```"
    assert extract.lcb_code(reply) == "second()"
    assert extract.lcb_code("no fences") == ""


@pytest.mark.parametrize("reply,want", [
    ("... so the answer is (C).", "C"),
    ("Reasoning.\nAnswer: B", "B"),
    ("I pick J", "J"),
    ("no letter here", None),
])
def test_mmlu_pro_letter_chain(reply, want):
    assert extract.mmlu_pro_letter(reply) == want


def test_gpqa_letter_takes_the_last_answer_line():
    assert extract.gpqa_letter("Answer: A is tempting, but...\nAnswer: C") == "C"
    assert extract.gpqa_letter("nothing") is None


def test_bird_sql_prefers_sql_fence_and_strips_think():
    reply = "<think>\nhmm\n</think>\n```sql\nSELECT 1;\n```"
    assert extract.bird_sql(reply) == "SELECT 1;"
    assert extract.bird_sql("SELECT 2") == "SELECT 2"


def test_bfcl_calls_decodes_and_reports_bad_json():
    good = [{"function": {"name": "f", "arguments": '{"a": 1}'}}]
    assert extract.bfcl_calls(good) == ([{"f": {"a": 1}}], "")
    decoded, why = extract.bfcl_calls([{"function": {"name": "f", "arguments": '{"a": '}}])
    assert decoded is None and "JSON" in why
    assert extract.bfcl_calls([]) == ([], "")


# --- statistics --------------------------------------------------------------------------------


def test_mcnemar_exact_known_values():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert stats.mcnemar_exact(3, 7) == stats.mcnemar_exact(7, 3)
    assert stats.mcnemar_exact(5, 5) == 1.0


def test_paired_counts_only_discordant_items():
    base = {"a": True, "b": True, "c": False, "d": False}
    adapter = {"a": True, "b": False, "c": True, "d": True, "extra": True}
    p = stats.paired("ds1000", base, adapter)
    assert (p.n, p.losses, p.gains) == (4, 1, 2)
    assert p.acc_base == 50.0 and p.acc_adapter == 75.0 and p.delta == 25.0
    assert p.ci_low < p.delta < p.ci_high


def test_verdicts_follow_the_adr_thresholds():
    def mk(bench, delta, lo, p=0.5):
        return stats.Paired(bench, 500, 50.0, 50.0 + delta, delta, lo, delta + 1, 0, 0, p)

    assert stats.verdict(mk("ds1000", 3.0, 1.0, p=0.01)) == "gain"
    assert stats.verdict(mk("ds1000", 3.0, -1.0, p=0.2)) == "no gain"
    assert stats.verdict(mk("bird", -9.0, -11.0, p=1e-9)) == "regression"
    assert stats.verdict(mk("bird", -2.0, -5.0, p=0.3)) == "no gain"
    assert stats.verdict(mk("ifeval", -1.5, -3.0)) == "fail (n.s.)"
    assert stats.verdict(mk("ifeval", -1.5, -3.0, p=0.01)) == "fail"
    assert stats.verdict(mk("ifeval", -0.5, -2.5)) == "pass (unresolved)"
    assert stats.verdict(mk("lcb", -0.5, -1.9)) == "pass (confirmed)"
    decision = stats.gate2_decision({"ds1000": mk("ds1000", 3.0, 1.0, p=0.01),
                                     "bird": mk("bird", -9.0, -11.0, p=1e-9),
                                     "ifeval": mk("ifeval", -1.5, -3.0)})
    assert decision["target_gains"] == ["ds1000"] and not decision["target_criterion_met"]
    assert decision["target_regressions"] == ["bird"]
    assert decision["regressions"] == ["ifeval"] and "gpqa" in decision["not_measured"]


# --- item building -----------------------------------------------------------------------------


def test_stratified_is_deterministic_and_capped():
    rows = [{"k": i % 3, "i": i} for i in range(30)]
    a = prepare.stratified(rows, lambda r: r["k"], 4)
    assert a == prepare.stratified(rows, lambda r: r["k"], 4)
    assert len(a) == 12 and all(sum(r["k"] == k for r in a) == 4 for k in range(3))
    assert prepare.stratified(rows, lambda r: r["k"], None) == rows


def test_bfcl_tools_match_bfcl_function_calling_conversion():
    fn = {"name": "math.factorial", "description": "Factorial.",
          "parameters": {"type": "dict", "required": ["n"], "properties": {
              "n": {"type": "integer", "description": "n"},
              "scale": {"type": "float", "description": "s"},
              "opts": {"type": "array", "items": {"type": "dict", "properties": {
                  "k": {"type": "tuple", "description": "k"}}}}}}}
    (tool,) = prepare.bfcl_tools([fn])
    f = tool["function"]
    assert f["name"] == "math_factorial" and f["parameters"]["type"] == "object"
    assert f["parameters"]["properties"]["scale"] == {
        "type": "number", "format": "float", "description": "s This is a float type value."}
    assert f["parameters"]["properties"]["opts"]["items"]["type"] == "object"
    assert f["parameters"]["properties"]["opts"]["items"]["properties"]["k"]["type"] == "array"
    assert fn["name"] == "math.factorial"  # the checker's copy is left untouched


def test_lcb_private_tests_decode_without_allowing_pickle_globals():
    tests = [{"input": "1\n", "output": "2\n", "testtype": "stdin"}]
    blob = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
    assert prepare._lcb_private(blob) == tests
    evil = base64.b64encode(zlib.compress(pickle.dumps(ValueError("x")))).decode()
    with pytest.raises(pickle.UnpicklingError):
        prepare._lcb_private(evil)


def test_items_roundtrip(tmp_path):
    item = Item("ifeval", "1", [{"role": "user", "content": "hi"}], {"max_tokens": 5},
                {"key": 1}, {"x": 1})
    save_items(tmp_path / "i.jsonl", [item])
    assert load_items(tmp_path / "i.jsonl") == [item]


# --- requests ----------------------------------------------------------------------------------


def test_request_body_sends_messages_and_limits_but_never_the_reference():
    item = Item("ds1000", "7", [{"role": "user", "content": "q"}],
                {"max_tokens": 1024, "stop": ["</code>"]}, {"reference_code": "SECRET"})
    base, adapter = request_body(item, "base", "m"), request_body(item, "adapter", "m")
    assert "SECRET" not in json.dumps(base)
    assert base["lora"] == [{"id": 0, "scale": 0.0}] and adapter["lora"][0]["scale"] == 1.0
    assert base["temperature"] == 0.0 and base["chat_template_kwargs"] == {
        "enable_thinking": False}
    assert base["stop"] == ["</code>"] and "stream" not in base
    tool_item = Item("bfcl", "s0", [{"role": "user", "content": "q"}], {"tools": [{"x": 1}]})
    assert request_body(tool_item, "base", "m")["stream"] is True


# --- contamination -----------------------------------------------------------------------------


def test_contamination_flags_overlap_by_level(tmp_path):
    words = " ".join(f"w{i}" for i in range(40))
    leaked = Item("mmlu_pro", "1", [{"role": "user", "content":
                                     f"Question: {words}\nAnswer: Let's think"}])
    clean = Item("mmlu_pro", "2", [{"role": "user", "content":
                                    "Question: " + " ".join(f"z{i}" for i in range(40))
                                    + "\nAnswer: Let's think"}])
    mix = tmp_path / "mix.jsonl"
    mix.write_text(json.dumps({"messages": [{"role": "user", "content": words}]}) + "\n")
    result = contamination.scan([leaked, clean], [mix])["mmlu_pro"]
    assert result == {"n": 2, "any": ["1"], "strong": ["1"]}


# --- report ------------------------------------------------------------------------------------


def test_report_drops_unmeasurable_items_and_reads_the_aa_control(tmp_path):
    from dsbench.battery import report
    from dsbench.battery.items import write_jsonl

    items = [Item("ds1000", str(i), [{"role": "user", "content": "q"}], meta={"library": "L"})
             for i in range(6)]
    save_items(tmp_path / "items" / "ds1000.jsonl", items)
    outcomes = {"gold": [1, 1, 1, 1, 1, 0], "base": [1, 1, 0, 0, 0, 1],
                "adapter": [1, 1, 1, 1, 0, 0], "base_rep": [1, 0, 0, 0, 0, 1]}
    for state, passed in outcomes.items():
        write_jsonl(tmp_path / "scores" / f"ds1000.{state}.jsonl",
                    [{"id": str(i), "passed": bool(p), "status": "ok"}
                     for i, p in enumerate(passed)])
    s = report.bench_summary(tmp_path, "ds1000", strong={"3"})
    assert s["unmeasurable"] == ["5"]  # gold fails: dropped before pairing
    assert (s["paired"]["n"], s["paired"]["gains"], s["paired"]["losses"]) == (5, 2, 0)
    assert s["aa"] == {"flips": 1, "n": 5, "acc_base_rep": 20.0}
    assert s["clean"]["n"] == 4 and s["clean"]["gains"] == 1  # item 3 (seen in training) removed
    assert s["strata"]["L"] == {"n": 5, "base": 40.0, "adapter": 80.0}


def test_dsbench_suite_scores_a_problem_by_majority_of_its_runs(tmp_path):
    from dsbench.battery import dsbench_suite, report

    def run(state, outcomes):
        results = [{"id": pid, "category": "de", "passed": p, "status": "ok" if p else "wrong"}
                   for pid, ps in outcomes.items() for p in ps]
        path = tmp_path / f"{state}.json"
        path.write_text(json.dumps({"meta": {}, "results": results}))
        return dsbench_suite.convert(path, state, tmp_path)

    base = run("base", {"a": [True, False, False], "b": [True, True, False]})
    run("adapter", {"a": [True, True, False], "b": [True, True, True]})
    assert [r["passed"] for r in base] == [False, True]
    s = report.bench_summary(tmp_path, "dsbench", strong=None)
    assert (s["paired"]["n"], s["paired"]["gains"], s["paired"]["losses"]) == (2, 1, 0)


class _LoraServer:
    """llama-server's /lora-adapters and /slots, as its handlers behave: a POST of scales sets the
    listed ids and puts every unlisted one at 0; a GET lists every loaded adapter with its scale;
    POST /slots/N?action=erase empties slot N's cache."""

    def __init__(self, n: int, slots: int = 8):
        self.scales = [1.0] * n  # --lora-init-without-apply leaves them reported at 1.0
        self.slots, self.erased = slots, []

    def handle(self, request):
        import httpx

        path = request.url.path
        if path.startswith("/slots"):
            if request.method == "GET":
                return httpx.Response(200, json=[{"id": i} for i in range(self.slots)])
            assert request.url.params["action"] == "erase"
            self.erased.append(int(path.rsplit("/", 1)[1]))
            return httpx.Response(200, json={"id_slot": self.erased[-1], "n_erased": 0})
        if request.method == "POST":
            body = json.loads(request.content)
            self.scales = [0.0] * len(self.scales)
            for entry in body:
                self.scales[entry["id"]] = entry["scale"]
            self.erased = []  # what a switch must erase is counted from here
            return httpx.Response(200, json={"success": True})
        return httpx.Response(200, json=[{"id": i, "path": f"a{i}.gguf", "scale": s}
                                         for i, s in enumerate(self.scales)])


def test_set_scale_drives_one_adapter_or_each_of_two(monkeypatch):
    import httpx
    from dsbench.battery import dsbench_suite

    real = httpx.Client  # before any patch: dsbench_suite.httpx is the httpx module itself

    def serve(server):
        monkeypatch.setattr(dsbench_suite.httpx, "Client",
                            lambda **kw: real(transport=httpx.MockTransport(server.handle), **kw))

    one = _LoraServer(1)
    serve(one)
    assert [a["scale"] for a in dsbench_suite.set_scale("http://x", 1.0)] == [1.0]
    # Every slot's cache goes with the switch: pi's next request can't reuse the old scale's KV.
    assert one.erased == list(range(8))
    two = _LoraServer(2)
    serve(two)
    # Revision 2 alone on a server that loaded Revision 2.1 first: id 0 at 0, id 1 at 1.
    assert [a["scale"] for a in dsbench_suite.set_scale("http://x", [0.0, 1.0])] == [0.0, 1.0]
    assert two.scales == [0.0, 1.0] and two.erased == list(range(8))
    # One scale on a two-adapter server is refused (the read lists two), before any erase.
    with pytest.raises(RuntimeError, match="wanted scales"):
        dsbench_suite.set_scale("http://x", 1.0)
    assert two.erased == []


def test_report_separates_think_leak_from_capability(tmp_path):
    from dsbench.battery import report
    from dsbench.battery.items import write_jsonl

    items = [Item("humaneval_plus", str(i), [{"role": "user", "content": "q"}]) for i in range(4)]
    save_items(tmp_path / "items" / "humaneval_plus.jsonl", items)
    contents = {"base": ["code"] * 4,
                "adapter": ["<think>\nlong reasoning, no end", "<think>\nr\n</think>\n\ncode",
                            "code", "code"]}
    passed = {"base": [1, 1, 1, 0], "adapter": [0, 1, 1, 1]}
    for state in ("base", "adapter"):
        write_jsonl(tmp_path / "gen" / f"humaneval_plus.{state}.jsonl",
                    [{"id": str(i), "content": c} for i, c in enumerate(contents[state])])
        write_jsonl(tmp_path / "scores" / f"humaneval_plus.{state}.jsonl",
                    [{"id": str(i), "passed": bool(p), "status": "ok"}
                     for i, p in enumerate(passed[state])])
    s = report.bench_summary(tmp_path, "humaneval_plus", strong=None)
    assert s["think_leak"] == {"base": {"none": 4}, "adapter": {"closed": 1, "none": 2, "open": 1}}
    assert (s["paired"]["gains"], s["paired"]["losses"]) == (1, 1)
    assert s["no_leak"]["n"] == 2 and (s["no_leak"]["gains"], s["no_leak"]["losses"]) == (1, 0)


def test_exact_interval_agrees_with_the_exact_test():
    # Reference Clopper-Pearson values (R: binom.test(4, 16)$conf.int = 0.0727, 0.5238).
    low, high = stats.clopper_pearson(4, 16)
    assert low == pytest.approx(0.07266, abs=1e-4) and high == pytest.approx(0.52377, abs=1e-4)
    base = {str(i): i < 12 for i in range(163)}  # 12 losses, 4 gains, as HumanEval+ gave
    adapter = {str(i): 12 <= i < 16 for i in range(163)}
    p = stats.paired("humaneval_plus", base, adapter)
    assert p.p > 0.05 and p.ci_low < 0 < p.ci_high  # test and interval agree: no significance
    assert stats.clopper_pearson(0, 10)[0] == 0.0 and stats.clopper_pearson(10, 10)[1] == 1.0


def test_half_scale_state_is_paired_with_base(tmp_path):
    from dsbench.battery import report
    from dsbench.battery.items import STATE_SCALE, write_jsonl

    assert STATE_SCALE["adapter_half"] == 0.5
    item = Item("ifeval", "1", [{"role": "user", "content": "q"}])
    assert request_body(item, "adapter_half", "m")["lora"] == [{"id": 0, "scale": 0.5}]
    save_items(tmp_path / "items" / "ifeval.jsonl",
               [Item("ifeval", str(i), [{"role": "user", "content": "q"}]) for i in range(3)])
    outcomes = {"base": [1, 1, 0], "adapter": [0, 0, 0], "adapter_half": [1, 1, 1]}
    for state, passed in outcomes.items():
        rows = [{"id": str(i), "passed": bool(p), "status": "ok"} for i, p in enumerate(passed)]
        write_jsonl(tmp_path / "scores" / f"ifeval.{state}.jsonl", rows)
    s = report.bench_summary(tmp_path, "ifeval", strong=None)
    assert s["paired"]["losses"] == 2 and s["half"]["paired"]["gains"] == 1
    half = report.half_decision({"ifeval": s})
    assert half["regressions"] == [] and "mmlu_pro" in half["not_measured"]
    md = report.markdown({"ifeval": s}, stats.gate2_decision({}), half)
    assert "| ifeval | 3 | 66.7 | 0.0 | 100.0 | +33.3" in md and "At scale 0.5:" in md


def test_bfcl_is_reported_as_its_leaderboard_columns(tmp_path):
    from dsbench.battery import report
    from dsbench.battery.items import write_jsonl

    cats = ["simple", "simple", "irrelevance", "irrelevance"]
    save_items(tmp_path / "items" / "bfcl.jsonl",
               [Item("bfcl", str(i), [{"role": "user", "content": "q"}], meta={"category": c})
                for i, c in enumerate(cats)])
    outcomes = {"base": [0, 1, 1, 1], "adapter": [1, 1, 0, 0]}
    for state, passed in outcomes.items():
        rows = [{"id": str(i), "passed": bool(p), "status": "ok", "extra": {"format_ok": True}}
                for i, p in enumerate(passed)]
        write_jsonl(tmp_path / "scores" / f"bfcl.{state}.jsonl", rows)
    bench, only = report.SPLITS["bfcl_ast"]
    ast = report.bench_summary(tmp_path, bench, None, row="bfcl_ast", only=only)
    bench, only = report.SPLITS["bfcl_irrelevance"]
    irr = report.bench_summary(tmp_path, bench, None, row="bfcl_irrelevance", only=only)
    assert (ast["paired"]["n"], ast["paired"]["gains"], ast["paired"]["losses"]) == (2, 1, 0)
    assert (irr["paired"]["n"], irr["paired"]["gains"], irr["paired"]["losses"]) == (2, 0, 2)
    assert "bfcl_ast" in stats.TARGET_BENCHES


def test_significant_change_outside_the_adr_list_is_flagged():
    worse = stats.Paired("bfcl_irrelevance", 240, 89.0, 67.0, -22.0, -25.0, -17.0, 57, 4, 1e-12)
    assert stats.verdict(worse) == "regression (no ADR limit)"
    flat = stats.Paired("bfcl_irrelevance", 240, 89.0, 88.0, -1.0, -3.0, 1.0, 5, 3, 0.7)
    assert stats.verdict(flat) == "info"


def test_parity_checks_each_adapter_and_that_scale_1_takes_nothing_from_cache(tmp_path):
    from dsbench.battery import parity

    n = len(parity.PROMPTS)
    base = [f"base {i}" for i in range(n)]
    rev21 = [f"rev2.1 {i}" if i < 2 else base[i] for i in range(n)]  # changes 2 of 8
    rev2 = [f"rev2 {i}" if i < 4 else base[i] for i in range(n)]  # changes 4 of 8

    def steps(scale1):
        return {"0:0.0": base, "1:1.0": scale1, "2:0.0": base}

    def cached(scale1):
        # The second scale-0 step may reuse the first one's KV: the same state.
        return {"0:0.0": [0] * n, "1:1.0": scale1, "2:0.0": [30] * n}

    run = tmp_path / "parity.json"
    clean = {"nolora": {"none": base}, "lora": steps(rev21), "lora_cached": cached([0] * n),
             "lora2": steps(rev2), "lora2_cached": cached([0] * n)}
    run.write_text(json.dumps(clean))
    result = parity.compare(run)
    assert (result["scale0_equals_nolora"], result["scale1_differs_from_scale0"]) == (n, 2)
    assert result["lora2"]["scale1_differs_from_scale0"] == 4
    assert result["scale1_tokens_from_cache"] == 0 and parity.passed(result)
    # Every window before 2026-10-09: prompts 1, 2 and 6 began their scale-1 replies from their
    # own scale-0 KV, loaded from the server's RAM copy. The replies can't show it; the counts do.
    run.write_text(json.dumps({**clean, "lora2_cached": cached([0, 30, 19, 0, 0, 0, 17, 0])}))
    leaked = parity.compare(run)
    assert leaked["lora2"]["scale1_tokens_from_cache"] == 66 and not parity.passed(leaked)
    # A file without the counts (every one before 2026-10-09) can't show the scales apart.
    run.write_text(json.dumps({"nolora": {"none": base}, "lora": steps(rev21)}))
    old = parity.compare(run)
    assert "lora2" not in old and old["scale1_tokens_from_cache"] is None
    assert not parity.passed(old)


def test_parity_requests_name_one_adapter_and_record_the_cached_tokens(monkeypatch):
    import httpx
    from dsbench.battery import parity

    real, sent = httpx.Client, []

    def handle(request):
        sent.append(json.loads(request.content).get("lora"))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "reply"}}],
            "usage": {"prompt_tokens": 30, "prompt_tokens_details": {"cached_tokens": 26}}})

    monkeypatch.setattr(parity.httpx, "Client",
                        lambda **kw: real(transport=httpx.MockTransport(handle), **kw))
    replies, cached = parity.run("http://x", 1.0, lora_id=1)
    assert (replies, cached) == (["reply"] * len(parity.PROMPTS), [26] * len(parity.PROMPTS))
    assert sent[0] == [{"id": 1, "scale": 1.0}]  # the server puts adapter 0 at 0
    parity.run("http://x", None)
    assert sent[-1] is None  # the bare-base server is asked for no adapter
