"""Tests for the agent's own ClickHouse login (agentic/access.py) and how the harnesses use it, with
ClickHouse, the model and pi replaced. The live half, what a login can and can't read, runs in both
oracle gates (`dsbench-agent-selftest`, `python -m dsbench.sftgen.probe.tasks`)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from dsbench.agentic import access, audit, loop, pi_runner
from dsbench.agentic import tools as T
from dsbench.agentic.loader import load_problems
from dsbench.agentic.schema import AgentProblem, GradeContext
from dsbench.sftgen.ml_tasks import ML_TASKS
from dsbench.sftgen.probe import runner as probe_runner
from dsbench.sftgen.probe.tasks import PROBE_TASKS

LOGIN = access.AgentLogin("dsbench_agent_prob_x", "pw", "prob_x",
                          withheld=("x_test_full", "x_test_key"))


# ---- the login ----

def test_the_login_grants_its_database_then_takes_the_labels_back():
    statements, policies = access.login_statements(
        "u", "pw", "prob_x", withheld=("x_test_full", "x_test_key"),
        withheld_rows=(("aviation.flights", "Origin = 'ATL'"),))
    assert statements == [
        "DROP USER IF EXISTS u",
        "CREATE USER u IDENTIFIED WITH sha256_password BY 'pw' DEFAULT DATABASE prob_x",
        "GRANT ALL ON prob_x.* TO u",
        "REVOKE ALL ON prob_x.x_test_full FROM u",  # partial revokes: after the grant
        "REVOKE ALL ON prob_x.x_test_key FROM u",
        "GRANT SELECT ON aviation.* TO u",
        "CREATE ROW POLICY OR REPLACE u_hide_0 ON aviation.flights AS RESTRICTIVE FOR SELECT "
        "USING NOT (Origin = 'ATL') TO u",
    ]
    assert policies == (("u_hide_0", "aviation.flights"),)
    plain, _ = access.login_statements("u", "pw", "probe_x", shared=None)
    assert not any("aviation" in s for s in plain)  # nothing shared, nothing granted


def test_the_env_points_the_agents_python_at_its_login():
    assert LOGIN.env == {"CLICKHOUSE_USER": "dsbench_agent_prob_x", "CLICKHOUSE_PASSWORD": "pw",
                         "CLICKHOUSE_DB": "prob_x"}


def test_every_table_a_setup_writes_is_an_input_or_withheld():
    # A table the prompt names is the agent's input. Any other is the grader's: it holds the test
    # rows' labels, and only its name keeps it from the agent's login. Target C's tasks
    # (ml_tasks) run under the same login, in the generator.
    for problem in load_problems() + PROBE_TASKS + ML_TASKS:
        for table in audit.setup_tables(problem):
            named = re.search(rf"\b{table}\b", problem.prompt) is not None
            withheld = access.WITHHELD_TABLE.search(table) is not None
            assert named != withheld, (problem.id, table, named, withheld)


def test_problems_built_from_the_warehouse_withhold_their_test_rows():
    by_id = {p.id: p for p in load_problems()}
    for pid in ("ds_cancel_predict", "ds_delay_predict", "ds_taxi_regression"):
        [(table, condition)] = by_id[pid].withheld_rows
        assert table == "aviation.flights" and condition.endswith("% 100 < 8"), pid
    assert by_id["ds_notam_classify"].withheld_rows == (("aviation.notam", "split = 'test'"),)
    # Every other problem leaves the warehouse whole: its oracle and the agent see the same rows.
    assert all(not p.withheld_rows for p in by_id.values() if not p.id.startswith("ds_"))


def test_no_prompt_hands_the_agent_the_admins_login():
    for prompt in (pi_runner.SYSTEM_PI, probe_runner.NEUTRAL_SYSTEM, loop.SYSTEM):
        assert "avbench" not in prompt
    assert "os.environ['CLICKHOUSE_USER']" in pi_runner.SYSTEM_PI
    assert "os.environ['CLICKHOUSE_PASSWORD']" in probe_runner.NEUTRAL_SYSTEM
    wrapper = (Path(pi_runner._BIN) / "run_sql").read_text()
    assert "user=avbench" not in wrapper and "X-ClickHouse-User: ${CLICKHOUSE_USER" in wrapper


# ---- the guard ----

def _native(*calls):
    return [{"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": json.dumps(args)}} for name, args in calls]}]


def _pi(*commands):
    return [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "cancel_test_key would tell me"},  # thinking isn't a call
        *({"type": "toolCall", "name": "bash", "arguments": {"command": c}} for c in commands)]}]


@pytest.mark.parametrize(("trajectory", "why"), [
    (_native(("run_sql", {"query": "SELECT * FROM x_test_key"})),
     "named withheld table(s): x_test_key"),
    (_pi("run_sql \"SELECT id, cancelled FROM prob_x.x_test_full\""),
     "named withheld table(s): x_test_full"),
    (_pi("curl 'http://localhost:8123/?user=avbench&password=avbench' -d 'SELECT 1'"),
     "used the sandbox admin's login"),
    (_native(("run_python", {"code": "get_client(username='avbench', password='avbench')"})),
     "used the sandbox admin's login"),
])
def test_a_run_reaching_for_the_labels_is_caught(trajectory, why):
    assert access.breach(trajectory, LOGIN.withheld) == why


@pytest.mark.parametrize("trajectory", [
    _native(("run_python", {"code": "X_test_full = np.vstack([X_test, extra])"})),  # its own array
    _native(("run_sql", {"query": "SHOW TABLES"})),
    _pi("docker exec avbench-workspace ls", "run_sql \"SELECT count() FROM x_test\""),
    _pi(),  # only thinking: never an action
])
def test_an_honest_run_is_left_alone(trajectory):
    assert access.breach(trajectory, LOGIN.withheld) is None


# ---- the harnesses ----

def _problem(passes=True):
    return AgentProblem(id="ds_x", category="ds", difficulty="hard", title="t", prompt="p",
                        check=lambda ctx: (passes, "graded"), reference=lambda ctx: None,
                        withheld_rows=(("aviation.flights", "1"),))


def _fake_login(monkeypatch, module):
    events = []

    def open_login(ctx, withheld_rows=()):
        events.append(("open", withheld_rows))
        return LOGIN

    monkeypatch.setattr(module, "open_login", open_login)
    monkeypatch.setattr(module, "close_login", lambda login: events.append(("close", login.user)))
    return events


def _replies(*calls):
    out = [{"message": {"content": "", "tool_calls": [
        {"id": "c", "function": {"name": n, "arguments": json.dumps(a)}}]}} for n, a in calls]
    return out + [{"message": {"content": "", "tool_calls": [
        {"id": "f", "function": {"name": "finish", "arguments": "{}"}}]}}]


def _native_run(monkeypatch, replies, passes=True):
    events = _fake_login(monkeypatch, loop)
    used = {}
    monkeypatch.setattr(access.AgentLogin, "client", lambda self: "agent-client")
    monkeypatch.setattr(loop, "call_model", lambda messages, **kw: replies.pop(0))
    monkeypatch.setattr(T, "run_sql", lambda client, q: used.setdefault("sql", client) and "1")
    monkeypatch.setattr(T, "run_python",
                        lambda code, ns, env=None: used.setdefault("env", env) and "ok")
    res = loop.run_agent(_problem(passes), GradeContext("admin-client", "prob_x"),
                         base_url="http://x", model="m")
    return res, used, events


def test_the_native_loop_runs_the_agent_as_its_login(monkeypatch):
    res, used, events = _native_run(monkeypatch, _replies(
        ("run_sql", {"query": "SELECT 1"}), ("run_python", {"code": "print(1)"})))
    assert (res.passed, res.status) == (True, "ok")
    assert used == {"sql": "agent-client", "env": LOGIN.env}  # never the grader's admin client
    assert events == [("open", (("aviation.flights", "1"),)), ("close", LOGIN.user)]


def test_the_native_loop_fails_a_run_that_named_a_withheld_table(monkeypatch):
    res, _, events = _native_run(monkeypatch, _replies(
        ("run_sql", {"query": "SELECT * FROM x_test_key"})), passes=True)
    assert (res.passed, res.status, res.reason) == (
        False, "withheld_access", "named withheld table(s): x_test_key")
    assert events[-1] == ("close", LOGIN.user)


def test_pi_runs_with_the_logins_credentials_and_is_guarded(monkeypatch, tmp_path):
    events = _fake_login(monkeypatch, pi_runner)
    seen = {}
    messages = _pi("run_sql \"SELECT * FROM x_test_full\"")

    class Done:
        returncode = 0
        stderr = ""
        stdout = "\n".join(json.dumps(e) for e in (
            {"type": "turn_end"}, {"type": "agent_end", "messages": messages},
            {"type": "agent_settled"}))

    def fake_run(cmd, cwd, env, **kw):
        seen.update(user=env["CLICKHOUSE_USER"], password=env["CLICKHOUSE_PASSWORD"],
                    db=env["CLICKHOUSE_DB"])
        return Done()

    monkeypatch.setattr(pi_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(pi_runner.tempfile, "mkdtemp", lambda prefix: str(tmp_path))
    res = pi_runner.run_pi_agent(_problem(passes=True), GradeContext("admin", "prob_x"),
                                 provider="p", model="m", thinking="high", timeout=5)
    assert seen == {"user": LOGIN.user, "password": "pw", "db": "prob_x"}
    assert (res.passed, res.status) == (False, "withheld_access")
    assert events == [("open", (("aviation.flights", "1"),)), ("close", LOGIN.user)]


# ---- the audit of runs from before the fix ----

def test_the_audit_flags_what_reached_for_the_labels_and_nothing_else():
    tables = ["delay_train", "delay_test_full", "delay_test", "delay_test_key"]

    def flags(pid, trajectory, outputs=""):
        traj = trajectory + ([{"role": "toolResult", "content": [{"type": "text",
                                                                    "text": outputs}]}]
                             if outputs else [])
        return audit.audit_result({"id": pid, "trajectory": traj}, tables)

    assert flags("ds_delay_predict", _pi("run_sql \"SELECT * FROM delay_test_full\""),
                 "delay_test_full") == {"named_withheld": ["delay_test_full"],
                                        "saw_withheld_listed": True}
    joined = _pi("run_python <<EOF\nSELECT t.id FROM delay_test t "
                 "JOIN aviation.flights f ON 1\nEOF")
    assert flags("ds_delay_predict", joined) == {"joined_test_to_flights": True}
    assert flags("ds_delay_predict", _pi("run_sql \"SELECT count() FROM aviation.flights\"")) == {
        "read_flights": True}
    counts = _pi("run_sql \"SELECT category, count() FROM aviation.notam GROUP BY category\"")
    assert flags("ds_notam_classify", counts) == {"notam_test_labels": ["aggregate"]}
    # The train split, its quotes escaped twice inside a JSON-encoded heredoc, is not a leak.
    train = _native(("run_python", {"code": "q('SELECT text, category FROM aviation.notam "
                                            "WHERE split=\\\\'train\\\\'')"}))
    assert flags("ds_notam_classify", train) == {}
    assert flags("ds_delay_predict", _native(("run_python", {"code": "X_test_full = 1"}))) == {}
