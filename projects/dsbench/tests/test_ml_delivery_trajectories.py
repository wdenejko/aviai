"""Tests for the Target C generator's loop (sftgen/ml_delivery_trajectories.py), with the model and
the sandbox's tools replaced: what a trajectory keeps, how a run ends, and how runs are ordered.
The live parts (llama-server, ClickHouse, the workspace container) run in the GPU window."""
from __future__ import annotations

import json

from dsbench.agentic.schema import AgentProblem, GradeContext
from dsbench.sftgen import ml_delivery_trajectories as mdt
from dsbench.sftgen.ml_tasks import ML_TASKS


def _reply(reasoning="", content="", calls=(), finish="tool_calls", tokens=50):
    tool_calls = [{"id": f"c{i}", "type": "function",
                   "function": {"name": name, "arguments": json.dumps(args)}}
                  for i, (name, args) in enumerate(calls)]
    message = {"role": "assistant", "content": content, "tool_calls": tool_calls}
    if reasoning:
        message["reasoning_content"] = reasoning
    return {"message": message, "finish_reason": finish,
            "usage": {"prompt_tokens": 100, "completion_tokens": tokens}, "timings": {}}


def _problem(passes=True, max_steps=5):
    return AgentProblem(id="mlc_test", category="ds", difficulty="hard", title="t",
                        prompt="Predict and write t_pred.", check=lambda ctx: (passes, "graded"),
                        reference=lambda ctx: None, max_steps=max_steps)


def _run(monkeypatch, replies, **kw):
    sent, ran = [], []

    def fake_call(messages, **call_kw):
        sent.append(([dict(m) for m in messages], call_kw))
        return replies[len(sent) - 1]

    monkeypatch.setattr(mdt, "_call", fake_call)
    monkeypatch.setattr(mdt.T, "run_python",
                        lambda code, ns, workdir=None: ran.append((code, ns, workdir)) or "ok")
    monkeypatch.setattr(mdt.T, "run_sql", lambda client, q: "1 row")
    res = mdt.run_teacher_agent(kw.pop("problem", _problem()), GradeContext(None, "ns_1"),
                                base_url="http://x", model="base", thinking=True,
                                workdir="/tmp/sftc/ns_1", seed=7, **kw)
    return res, sent, ran


def test_every_assistant_turn_keeps_its_reasoning_and_the_finish_call(monkeypatch):
    res, sent, ran = _run(monkeypatch, [
        _reply("Load the data first.", calls=[("run_python", {"code": "print(1)"})]),
        _reply("Table written, done.", calls=[("finish", {})]),
    ])
    assert (res["passed"], res["status"], res["steps"]) == (True, "ok", 2)
    roles = [m["role"] for m in res["trajectory"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert [m.get("reasoning_content") for m in res["trajectory"] if m["role"] == "assistant"] == [
        "Load the data first.", "Table written, done."]
    # The next request carries the earlier turn's reasoning, as the training row will.
    assert sent[1][0][2]["reasoning_content"] == "Load the data first."
    assert ran == [("print(1)", "ns_1", "/tmp/sftc/ns_1")]  # in the run's own directory
    assert [t["completion_tokens"] for t in res["turns"]] == [50, 50]
    assert [c[1]["seed"] for c in sent] == [8, 9]  # the run's seed plus the step


def test_a_reply_in_text_ends_the_run_and_is_kept_as_the_last_turn(monkeypatch):
    res, _, _ = _run(monkeypatch, [_reply("All done.", content="Done.", finish="stop")])
    assert res["trajectory"][-1] == {"role": "assistant", "content": "Done.",
                                     "reasoning_content": "All done."}
    assert res["status"] == "ok"


def test_a_turn_cut_by_max_tokens_is_never_kept(monkeypatch):
    res, _, _ = _run(monkeypatch, [_reply("Thinking for ever", finish="length")])
    assert (res["passed"], res["status"]) == (False, "turn_cap")
    assert "oracle would have passed" in res["reason"]


def test_a_run_out_of_steps_is_a_budget_failure(monkeypatch):
    loop = [_reply("Again.", calls=[("run_sql", {"query": "SELECT 1"})])] * 2
    res, _, _ = _run(monkeypatch, loop, problem=_problem(passes=False, max_steps=2))
    assert (res["passed"], res["status"], res["steps"]) == (False, "budget", 2)


def test_the_request_turns_thinking_on_with_qwen_sampling():
    body = mdt.request_body([], model="base", sampling=mdt.BASE_SAMPLING, thinking=True,
                            max_tokens=8192, seed=3)
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert (body["temperature"], body["top_p"], body["top_k"]) == (0.6, 0.95, 20)
    assert (body["max_tokens"], body["seed"], body["stream"]) == (8192, 3, True)
    teacher = mdt.request_body([], model="ling")
    assert "chat_template_kwargs" not in teacher and teacher["temperature"] == 0.3


def test_runs_go_by_run_index_so_overlapping_runs_are_different_tasks():
    jobs = mdt.jobs_for(None, reps=2, run_offset=100)
    assert [ix for ix, _ in jobs] == [100] * len(ML_TASKS) + [101] * len(ML_TASKS)
    assert [t.id for _, t in jobs[: len(ML_TASKS)]] == [t.id for t in ML_TASKS]
    assert mdt._seed("mlc_widget_defect", 100) == mdt._seed("mlc_widget_defect", 100)


def test_a_base_run_is_recorded_as_such():
    res = {"trajectory": [{"role": "user", "content": "x"}], "reason": "ok", "status": "ok",
           "steps": 3, "tool_calls": 2, "latency_s": 9.0, "turns": [{"step": 1}]}
    rec = mdt._record(ML_TASKS[0], res, "base", 100, thinking=True, sampling=mdt.BASE_SAMPLING)
    prov = rec["meta"]["provenance"]
    assert (prov["method"], prov["thinking"], prov["sampling"]["temperature"]) == (
        "base-in-sandbox", True, 0.6)
    assert rec["meta"]["id"] == f"C-{ML_TASKS[0].id}-100" and rec["meta"]["turns"] == [{"step": 1}]
