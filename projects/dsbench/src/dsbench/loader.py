"""Discover problems by walking problems/<category>/*.py and collecting each module's PROBLEM.

Adding a task = dropping one file in problems/de|da|ds/. No registry to edit, nothing to import
by hand -- that is what keeps the set cheap to iterate on.
"""

from __future__ import annotations

import importlib
import pkgutil

from dsbench import problems as _problems_pkg
from dsbench.schema import CATEGORIES, Category, Difficulty, Problem


def load_problems(
    category: Category | None = None,
    difficulty: Difficulty | None = None,
    ids: set[str] | None = None,
) -> list[Problem]:
    """Return every PROBLEM under problems/, optionally filtered. Sorted by id for stable order."""
    found: dict[str, Problem] = {}
    for sub in CATEGORIES:
        pkg_name = f"{_problems_pkg.__name__}.{sub}"
        try:
            pkg = importlib.import_module(pkg_name)
        except ModuleNotFoundError:
            continue
        for mod in pkgutil.iter_modules(pkg.__path__):
            if mod.name.startswith("_"):
                continue
            module = importlib.import_module(f"{pkg_name}.{mod.name}")
            prob = getattr(module, "PROBLEM", None)
            if prob is None:
                continue
            if prob.id in found:
                raise ValueError(f"duplicate problem id {prob.id!r}")
            found[prob.id] = prob

    out = list(found.values())
    if category:
        out = [p for p in out if p.category == category]
    if difficulty:
        out = [p for p in out if p.difficulty == difficulty]
    if ids:
        out = [p for p in out if p.id in ids]
    return sorted(out, key=lambda p: p.id)


def module_path(problem_id: str) -> str:
    """Dotted module path for a problem id like 'de_easy_01' -> 'dsbench.problems.de.de_easy_01'."""
    category = problem_id.split("_", 1)[0]
    return f"{_problems_pkg.__name__}.{category}.{problem_id}"
