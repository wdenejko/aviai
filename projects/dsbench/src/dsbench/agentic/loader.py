"""Discover agentic problems by walking problems/<category>/*.py for each module's PROBLEM."""
from __future__ import annotations

import importlib
import pkgutil

from dsbench.agentic import problems as _pkg
from dsbench.agentic.schema import AgentProblem

CATEGORIES = ("de", "da", "ds")


def load_problems(category: str | None = None, ids: set[str] | None = None) -> list[AgentProblem]:
    found: dict[str, AgentProblem] = {}
    for sub in CATEGORIES:
        try:
            pkg = importlib.import_module(f"{_pkg.__name__}.{sub}")
        except ModuleNotFoundError:
            continue
        for mod in pkgutil.iter_modules(pkg.__path__):
            if mod.name.startswith("_"):
                continue
            module = importlib.import_module(f"{_pkg.__name__}.{sub}.{mod.name}")
            prob = getattr(module, "PROBLEM", None)
            if prob is None:
                continue
            if prob.id in found:
                raise ValueError(f"duplicate agentic problem id {prob.id!r}")
            found[prob.id] = prob
    out = list(found.values())
    if category:
        out = [p for p in out if p.category == category]
    if ids:
        out = [p for p in out if p.id in ids]
    return sorted(out, key=lambda p: p.id)
