"""Tests for the prompt-cache audit (battery/cache_audit.py): requests read off llama-server's log
lines, split into pi runs, matched to the run files, and a leak like the probe windows' found."""
from __future__ import annotations

import json

import pytest
from dsbench.battery import cache_audit as ca


def _lines(task: int, prompt: int, cached: int, gen: int) -> list[str]:
    """One finished request, as llama-server logs it."""
    return [
        f"0.01.000.000 I slot launch_slot_: id  0 | task {task} | processing task, is_child = 0",
        f"0.01.100.000 I slot print_timing: id  0 | task {task} | prompt eval time =     10.00 ms /"
        f"  {prompt - cached} tokens (    1.00 ms per token,  1000.00 tokens per second)",
        f"0.01.100.001 I slot print_timing: id  0 | task {task} |        eval time =     10.00 ms /"
        f"  {gen} tokens (   10.00 ms per token,   100.00 tokens per second)",
        f"0.01.100.002 I slot      release: id  0 | task {task} | stop processing: "
        f"n_tokens = {prompt + gen - 1}, truncated = 0",
    ]


def _log(tmp_path, requests, ram_off=True):
    """A window's server log: the parity check's bare-base server (ignored), then the server with
    the adapter, with the parity check's short prompts before pi's requests."""
    lines = ["0.00.000.001 I srv    load_model: loading model 'base.gguf'", *_lines(0, 30, 0, 9),
             "0.00.000.002 I srv    load_model: loading model 'base.gguf'"]
    if ram_off:
        lines.append("0.00.000.003 W srv          init: --cache-idle-slots requires --cache-ram,"
                     " disabling")
    for task, (prompt, cached, gen) in enumerate([(26, 0, 40), (34, 0, 40), *requests], start=1):
        lines += _lines(task, prompt, cached, gen)
    path = tmp_path / "server.log"
    path.write_text("\n".join(lines) + "\n")
    return path


def _runs(tmp_path, name, problems):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"meta": {"label": name},
                                "results": [{"id": p, "passed": True} for p in problems]}))
    return path


# Two steps of two problems, k = 2, each run two turns. A run's second turn reuses its first; a
# problem's second run reuses its first run's prefix (all but the last ~512 tokens, as the hybrid
# model's checkpoints allow). In step b, p2's first run starts from step a's prefix: the leak.
STEP_A = [(1700, 0, 100), (2600, 1650, 50), (1700, 1189, 80), (2600, 2100, 50),
          (1760, 0, 60), (2700, 1700, 40), (1760, 1250, 70), (2700, 2200, 40)]
STEP_B = [(1700, 0, 90), (2600, 1650, 50), (1700, 1189, 90), (2600, 2100, 50),
          (1760, 1250, 60), (2700, 1700, 40), (1760, 1250, 60), (2700, 2200, 40)]


def test_requests_and_runs_are_read_off_the_log(tmp_path):
    requests, ram_off = ca.server_requests(_log(tmp_path, STEP_A))
    assert ram_off and len(requests) == 2 + 8  # the bare-base server's request is not among them
    assert (requests[2].prompt, requests[2].cached) == (1700, 0)
    assert (requests[3].prompt, requests[3].cached) == (2600, 1650)
    runs = ca.pi_runs(requests)
    assert [len(r) for r in runs] == [2, 2, 2, 2]  # the parity check's prompts are left out


def test_a_problem_whose_first_run_starts_from_another_step_is_listed(tmp_path):
    log = _log(tmp_path, STEP_A + STEP_B)
    files = [_runs(tmp_path, "a", ["p1", "p1", "p2", "p2"]),
             _runs(tmp_path, "b", ["p1", "p1", "p2", "p2"])]
    result = ca.audit(log, files)
    assert result["steps"]["a"] == {"p1": [0, 1189], "p2": [0, 1250]}
    assert result["step_first_request_cached"] == {"a": 0, "b": 0}
    assert result["first_runs_from_cache"] == [{"step": "b", "problem": "p2", "cached": 1250}]
    # With the RAM copy off and every step starting empty, nothing of another step is reachable.
    assert result["clean"]
    # The probe windows: the RAM copy on, so b's p2 could, and did, load a's prefix.
    assert not ca.audit(_log(tmp_path, STEP_A + STEP_B, ram_off=False), files)["clean"]


def test_a_step_that_starts_from_cache_is_not_clean(tmp_path):
    first_from_cache = [(1700, 1189, 90), *STEP_B[1:]]  # the slots were not erased at the switch
    files = [_runs(tmp_path, "a", ["p1", "p1", "p2", "p2"]),
             _runs(tmp_path, "b", ["p1", "p1", "p2", "p2"])]
    result = ca.audit(_log(tmp_path, STEP_A + first_from_cache), files)
    assert result["step_first_request_cached"]["b"] == 1189 and not result["clean"]


def test_a_compaction_continues_its_run(tmp_path):
    # pi past its context limit: 9,000 tokens of conversation, then a 4,000-token summary prompt.
    compacted = [(1700, 0, 100), (9000, 2000, 50), (4000, 1189, 60), (4500, 3480, 40),
                 (1700, 1189, 80), (2600, 2100, 50)]
    runs = ca.pi_runs(ca.server_requests(_log(tmp_path, compacted))[0])
    assert [len(r) for r in runs] == [4, 2]
    assert [r[0].prompt for r in runs] == [1700, 1700]


def test_runs_that_do_not_match_the_run_files_are_refused(tmp_path):
    with pytest.raises(ValueError, match="don't match"):
        ca.audit(_log(tmp_path, STEP_A), [_runs(tmp_path, "a", ["p1", "p1", "p2"])])


def test_the_cli_exits_nonzero_unless_clean(tmp_path, monkeypatch, capsys):
    files = [_runs(tmp_path, "a", ["p1", "p1", "p2", "p2"])]
    for ram_off, code in ((True, 0), (False, 1)):
        monkeypatch.setattr("sys.argv", ["cache_audit", "--server-log",
                                         str(_log(tmp_path, STEP_A, ram_off=ram_off)),
                                         "--runs", *map(str, files)])
        with pytest.raises(SystemExit) as e:
            ca.main()
        assert e.value.code == code
    assert "NOT CLEAN" in capsys.readouterr().out
