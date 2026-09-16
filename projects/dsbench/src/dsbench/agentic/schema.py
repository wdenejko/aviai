"""The contract for an agentic problem — the v2 analogue of ADR-002's `Problem`.

Same discipline, adapted to a stateful, multi-step world:
  - `setup(ctx)`  prepares a per-problem scratch namespace (a dedicated ClickHouse database) so
                  problems never collide and re-runs are clean (ADR-003 §3).
  - the agent then operates the sandbox with tools (run_sql / run_python / finish).
  - `check(ctx)`  grades the FINAL STATE: it recomputes the expected answer INDEPENDENTLY from the
                  warehouse (never trusting the agent's output) and asserts against what the agent
                  produced — a scratch table it built and/or the value it finished with.
  - `reference(ctx)` is a correct solution that reaches the graded state; it powers the oracle gate
                  (`dsbench-agent-selftest`): if the reference doesn't pass check(), the bug is in
                  the problem, not the model. reference and check must derive the answer by
                  DIFFERENT routes, or the gate proves nothing (same rule as v1).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GradeContext:
    """Everything a setup/reference/check needs to touch the stack for one problem."""

    client: Any  # clickhouse-connect client whose DEFAULT database is `namespace`
    namespace: str  # per-problem scratch DB, e.g. "prob_da_hub_delay"
    answer: Any = None  # the value the agent finished with (for answer-graded problems)


CheckFn = Callable[[GradeContext], "bool | tuple[bool, str]"]
RefFn = Callable[[GradeContext], Any]  # returns the answer for answer-graded problems, else None
SetupFn = Callable[[GradeContext], None]  # seed problem-specific tables into the fresh namespace


@dataclass(frozen=True)
class AgentProblem:
    id: str
    category: str  # de | da | ds
    difficulty: str  # easy | medium | hard | expert
    title: str
    prompt: str  # the task shown to the agent; states the exact deliverable
    check: CheckFn
    reference: RefFn
    # Optional: runs after the empty scratch namespace is created and BEFORE the agent (or the
    # reference) acts — use it to materialise inputs the task hands the agent, e.g. an unlabeled
    # test table. Both the agent run and the oracle gate call it, so they see identical inputs.
    setup: SetupFn | None = None
    max_steps: int = 16
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class AgentResult:
    id: str
    category: str
    difficulty: str
    passed: bool
    # ok | wrong | error | model_error | truncated | setup_error
    status: str
    reason: str = ""
    steps: int = 0
    tool_calls: int = 0
    answer: Any = None
    trajectory: list = field(default_factory=list)  # full message list, for debugging / re-grading
    latency_s: float = 0.0
