"""Tests for the Target C generator (sftgen/ml_delivery_trajectories.py), with the model and the
sandbox's tools replaced: what a trajectory keeps, how a run ends, how runs are ordered, which runs
become rows (`selection`), and how quota mode spends its runs. The live parts (llama-server,
ClickHouse, the workspace container) run in the GPU window."""
from __future__ import annotations

import json

import pytest
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


# ---- which runs become rows ----

def _res(passed=True, status="ok", served=4000, reasoning="Plan.", calls=(), tool_output="ok"):
    """A finished run as run_teacher_agent returns it: one tool turn, then `finish`."""
    def turn(name, args):
        return _assistant_turn(reasoning, [(name, args)])
    trajectory = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for name, args in calls:
        trajectory += [turn(name, args), {"role": "tool", "content": tool_output}]
    trajectory.append(turn("finish", {}))
    return {"passed": passed, "status": status, "reason": "graded", "trajectory": trajectory,
            "steps": len(calls) + 1, "tool_calls": len(calls) + 1, "latency_s": 1.0,
            "turns": [{"step": 1, "prompt_tokens": served - 40, "completion_tokens": 40}]}


def _assistant_turn(reasoning, calls):
    return mdt._assistant("", reasoning, [{"id": "c", "type": "function",
                                           "function": {"name": n, "arguments": json.dumps(a)}}
                                          for n, a in calls])


def test_a_clean_passing_run_that_fits_is_kept():
    sel = mdt.selection(_res(), block=8192, reasoning=True)
    assert (sel["kept"], sel["why_not"], sel["served_tokens"], sel["fits"]) == (
        True, None, 4000, True)


@pytest.mark.parametrize(("res", "why_not"), [
    (_res(passed=False, status="wrong"), "wrong"),
    (_res(reasoning=""), "empty_reasoning"),
    (_res(calls=[("run_sql", {"query": "SELECT * FROM widget_test_key"})]), "names_answer_key"),
    (_res(calls=[("run_python", {"code": "client.query_df('SELECT id, label FROM "
                                         "cust_test_key')"})]), "names_answer_key"),
    (_res(served=8192), "over_block"),  # the template adds a newline: 8,193 tokens
])
def test_a_run_is_not_kept_for_the_first_reason_that_applies(res, why_not):
    sel = mdt.selection(res, block=8192, reasoning=True)
    assert (sel["kept"], sel["why_not"]) == (False, why_not)


def test_the_block_edge_and_the_checks_that_depend_on_the_mode():
    assert mdt.selection(_res(served=8191), block=8192)["kept"]  # 8,192 with the newline
    assert mdt.selection(_res(served=50_000))["fits"] is None  # no block: length not checked
    assert mdt.selection(_res(reasoning=""), reasoning=False)["kept"]  # a teacher's row


def test_a_key_seen_in_a_listing_is_recorded_but_not_held_against_the_run():
    res = _res(calls=[("run_sql", {"query": "SHOW TABLES"})],
               tool_output="name\nwidget_test\nwidget_test_key\nwidget_train")
    sel = mdt.selection(res, block=8192, reasoning=True)
    assert (sel["kept"], sel["saw_answer_key"], sel["names_answer_key"]) == (True, True, False)


class _RecordingClient:
    """Enough of clickhouse_connect for a task's setup: the tables it creates."""

    def __init__(self):
        self.tables = []

    def command(self, sql):
        if sql.lstrip().startswith("CREATE TABLE"):
            self.tables.append(sql.split()[2].split(".")[-1])

    def insert_df(self, table, df):
        pass


@pytest.mark.parametrize("task", ML_TASKS, ids=lambda t: t.id)
def test_every_task_writes_one_answer_key_the_guard_catches(task):
    client = _RecordingClient()
    task.setup(GradeContext(client, "sftc_x_1"))
    keys = [t for t in client.tables if mdt.ANSWER_KEY.fullmatch(t)]
    assert len(keys) == 1, client.tables
    assert keys[0] not in task.prompt  # the agent is never told where the labels are
    assert not any(mdt.ANSWER_KEY.search(t) for t in client.tables if t != keys[0])


# ---- quota mode ----

def _quota(monkeypatch, outcome, *, quota=2, tasks=("mlc_widget_defect", "mlc_credit_leak"),
           **kw):
    """generate_quota with `outcome(task_id, run_ix) -> result` standing in for each run."""
    started = []

    def fake_run_job(task, run_ix, **job_kw):
        assert job_kw["gate"] is True  # every dataset passes its own oracle first
        started.append((task.id, run_ix))
        return outcome(task.id, run_ix)

    monkeypatch.setattr(mdt, "run_job", fake_run_job)
    kept, failed = [], []
    records, report = mdt.generate_quota(
        base_url="http://x", model="base", quota=quota, tasks=list(tasks), run_offset=200,
        thinking=True, sink=kept.append, fail_sink=failed.append, **kw)
    return started, kept, failed, report


def test_each_family_runs_until_it_has_its_rows(monkeypatch):
    # widget keeps every run; credit_leak keeps every third.
    def outcome(task_id, run_ix):
        third = (run_ix - 200) % 3 == 2
        return _res() if task_id == "mlc_widget_defect" or third else _res(served=12_000)

    started, kept, failed, report = _quota(monkeypatch, outcome, quota=2)
    by_task = {t: [ix for task, ix in started if task == t] for t, _ in started}
    assert by_task["mlc_widget_defect"] == [200, 201]  # stops at its quota
    assert by_task["mlc_credit_leak"] == [200, 201, 202, 203, 204, 205]  # 202 and 205 kept
    assert report["stop"] == "quotas filled"
    assert {f: v["kept"] for f, v in report["families"].items()} == {
        "mlc_widget_defect": 2, "mlc_credit_leak": 2}
    assert len(kept) == 4 and all(r["meta"]["selection"]["kept"] for r in kept)
    # The 4 over-block credit_leak runs passed the oracle and stay recoverable.
    assert [r["meta"]["selection"]["why_not"] for r in failed] == ["over_block"] * 4
    assert all(r["meta"]["verification"]["oracle_passed"] for r in failed)


def test_a_dataset_that_fails_its_own_oracle_uses_its_index_but_not_the_yield(monkeypatch):
    def outcome(task_id, run_ix):
        return (mdt._skipped("dataset_failed_oracle", "AP 0.149 < 0.15") if run_ix == 200
                else _res())

    started, kept, failed, report = _quota(monkeypatch, outcome, quota=1,
                                           tasks=("mlc_churn_rare",))
    assert started == [("mlc_churn_rare", 200), ("mlc_churn_rare", 201)]
    fam = report["families"]["mlc_churn_rare"]
    assert (fam["kept"], fam["skipped_datasets"], fam["runs_with_a_verdict"]) == (1, 1, 1)


def test_a_family_that_never_keeps_stops_at_max_runs(monkeypatch):
    started, kept, failed, report = _quota(
        monkeypatch, lambda t, ix: _res(passed=False, status="wrong"), quota=1,
        tasks=("mlc_energy_load",), max_runs=3)
    assert [ix for _, ix in started] == [200, 201, 202]
    assert report["stop"] == "max runs reached: mlc_energy_load"
    assert report["families"]["mlc_energy_load"]["shortfall"] == 1


def test_a_lost_server_stops_the_run_instead_of_using_up_datasets(monkeypatch):
    started, _, _, report = _quota(
        monkeypatch, lambda t, ix: mdt._skipped("model_error", "connection refused"),
        quota=5, max_runs=50)
    assert len(started) == 3 and report["stop"] == "3 consecutive model or harness errors"


def test_no_run_starts_after_the_time_limit(monkeypatch):
    now = iter(range(0, 10_000, 100))  # each clock read is 100 s later
    started, _, _, report = _quota(monkeypatch, lambda t, ix: _res(passed=False, status="wrong"),
                                   quota=5, max_seconds=250, clock=lambda: next(now))
    assert len(started) == 2 and report["stop"] == "time limit"


def test_resume_counts_earlier_rows_and_never_reuses_an_index(monkeypatch):
    # An earlier pass kept 1 widget row at 200 and failed at 203; 201-202 were lost in flight.
    prior = []
    for run_ix, res in ((200, _res()), (203, _res(passed=False, status="wrong"))):
        rec = mdt._record(ML_TASKS[0], res, "base", run_ix, thinking=True)
        rec["meta"]["selection"] = mdt.selection(res, block=8192, reasoning=True)
        prior.append(rec)
    started, kept, _, report = _quota(monkeypatch, lambda t, ix: _res(), quota=2,
                                      tasks=("mlc_widget_defect",), prior=prior)
    assert started == [("mlc_widget_defect", 204)] and len(kept) == 1
    assert report["prior"] == {"mlc_widget_defect": {"runs": 2, "kept": 1}}
    assert report["families"]["mlc_widget_defect"]["kept"] == 2


def test_parallel_runs_never_share_an_index_and_stop_near_the_quota(monkeypatch):
    def outcome(task_id, run_ix):
        return _res() if run_ix % 2 else _res(passed=False, status="wrong")

    started, kept, _, report = _quota(monkeypatch, outcome, quota=3, workers=4,
                                      tasks=[t.id for t in ML_TASKS])
    assert len(started) == len(set(started))
    for task in ML_TASKS:
        indices = sorted(ix for t, ix in started if t == task.id)
        assert indices == list(range(200, 200 + len(indices)))  # no gaps within a pass
        assert report["families"][task.id]["kept"] >= 3
    # Loops still running when a family fills its quota finish and are kept too, but the
    # scheduler counts them at the family's keep rate, so the overshoot stays small.
    assert len(kept) <= 7 * 3 + 4


def test_a_dataset_failing_the_gate_never_reaches_the_agent(monkeypatch):
    monkeypatch.setattr(mdt, "dataset_ok", lambda task, run_ix: (False, "AP 0.149 < 0.15"))
    monkeypatch.setattr(mdt, "run_teacher_agent", lambda *a, **k: pytest.fail("agent ran"))
    res = mdt.run_job(ML_TASKS[2], 100, base_url="http://x", model="base", gate=True)
    assert (res["status"], res["reason"]) == ("dataset_failed_oracle", "AP 0.149 < 0.15")
