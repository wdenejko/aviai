"""Tests for the pi harness glue that runs without pi: reading the per-reply cap a run records.

pi itself, the sandbox and the ClickHouse oracle are exercised by real runs (`dsbench-pi-run`).
"""

from __future__ import annotations

import json

from dsbench.agentic.pi_runner import pi_reply_cap


def _agent_dir(tmp_path, monkeypatch, models):
    (tmp_path / "models.json").write_text(json.dumps({"providers": models}))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))


def test_reply_cap_reads_the_models_max_tokens(tmp_path, monkeypatch):
    _agent_dir(tmp_path, monkeypatch, {"dashi-qwen36": {"apiKey": "x", "models": [
        {"id": "qwen36", "name": "Qwen3.6 (dashi)", "maxTokens": 12288}]}})
    assert pi_reply_cap("dashi-qwen36", "qwen36") == 12288
    assert pi_reply_cap("dashi-qwen36", "Qwen3.6 (dashi)") == 12288  # pi also matches the name


def test_reply_cap_falls_back_to_pis_default_when_the_model_sets_none(tmp_path, monkeypatch):
    _agent_dir(tmp_path, monkeypatch, {"dashi-qwen36": {"models": [{"id": "qwen36"}]}})
    assert pi_reply_cap("dashi-qwen36", "qwen36") == 16384


def test_reply_cap_is_unknown_outside_models_json(tmp_path, monkeypatch):
    _agent_dir(tmp_path, monkeypatch, {"dashi-qwen36": {"models": [{"id": "qwen36"}]}})
    assert pi_reply_cap("google", "gemini") is None  # a built-in provider: pi's catalogue decides
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "missing"))
    assert pi_reply_cap("dashi-qwen36", "qwen36") is None
