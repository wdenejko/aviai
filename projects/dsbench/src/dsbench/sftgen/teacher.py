"""Teacher client for Targets B and C -- a licence-clean model that authors reasoning prose.

The teacher NEVER supplies the label. The generator computes an execution-verified truth, asks the
teacher to solve the same task, and keeps the teacher's trace ONLY if its answer matches that truth
(ADR-004 execution-filtered distillation). So the teacher supplies phrasing; ground truth supplies
correctness, and provenance stays clean (teacher must be DeepSeek/Qwen/GLM/gpt-oss-class, ADR-001
B.4).

Two implementations:
  * `HTTPTeacher` -- an OpenAI-compatible chat client (httpx), for a local llama-server on dashi
    reached over the SSH tunnel, or any compatible endpoint.
  * `StubTeacher` -- offline, deterministic; takes a callable so tests can exercise BOTH the accept
    path (teacher returns the truth) and the reject path (teacher returns a wrong value) without a
    network or a GPU.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class TeacherReply:
    content: str  # the visible answer text (post-</think>)
    thinking: str | None  # the reasoning trace, if the model exposed one
    raw: str  # the full raw message content, for debugging


_THINK = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL)


def split_thinking(raw: str, reasoning_field: str | None = None) -> TeacherReply:
    """Separate a <think>...</think> block (or an API reasoning field) from the visible answer."""
    if reasoning_field:
        return TeacherReply(content=raw.strip(), thinking=reasoning_field.strip(), raw=raw)
    m = _THINK.search(raw)
    if m:
        return TeacherReply(content=m.group(2).strip(), thinking=m.group(1).strip(), raw=raw)
    return TeacherReply(content=raw.strip(), thinking=None, raw=raw)


class Teacher:
    def answer(self, system: str, user: str) -> TeacherReply:  # noqa: D102
        raise NotImplementedError


class HTTPTeacher(Teacher):
    """OpenAI-compatible chat client. Defaults suit a llama-server behind the dashi SSH tunnel."""

    def __init__(
        self, base_url: str = "http://localhost:18080/v1", model: str = "gpt-oss",
        api_key: str = "sk-noauth", temperature: float = 0.2, max_tokens: int = 2048,
        reasoning_effort: str | None = "high", timeout: float = 240.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout

    def answer(self, system: str, user: str) -> TeacherReply:
        import httpx

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort  # gpt-oss / harmony honours this
        r = httpx.post(
            f"{self.base_url}/chat/completions", json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout,
        )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        return split_thinking(msg.get("content") or "", msg.get("reasoning_content"))


class StubTeacher(Teacher):
    """Offline teacher: `fn(system, user) -> (thinking, content)`. For tests, not for real data."""

    def __init__(self, fn: Callable[[str, str], tuple[str | None, str]]) -> None:
        self._fn = fn

    def answer(self, system: str, user: str) -> TeacherReply:
        thinking, content = self._fn(system, user)
        return TeacherReply(content=content, thinking=thinking, raw=content)
