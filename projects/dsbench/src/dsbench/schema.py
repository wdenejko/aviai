"""The one contract every benchmark problem obeys.

Design goal: a problem must be *self-contained* and *execution-verified*. "Self-contained"
means the fixtures, the prompt shown to the model, the reference (gold) solution, and the
correctness check all live in one small module you can read top to bottom. "Execution-verified"
means we never grade free text or trust an LLM judge for the core signal -- we run the model's
code against the fixtures and assert on what it actually produced. That is the only kind of
score that stays honest across model iterations (see the ADR's eval section and avtext's
oracle-first methodology).

Two answer modes cover data-engineering / data-analysis / data-science work:

  mode="python"  the model returns a ```python block defining a function (default name
                 `solve`). We call it with the kwargs from `make_inputs()` and pass its
                 return value to `check()`.

  mode="sql"     the model returns a ```sql block with one SELECT. We register the tables
                 from `make_inputs()["tables"]` into an in-memory DuckDB, run the query, and
                 pass the resulting DataFrame to `check()`.

`make_inputs()` MUST be deterministic (fixed seeds, fixed data) so that `check()` can rebuild
the very same inputs to compute the expected answer -- the model process and the checker never
share memory, only the pickled result crosses the boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Category = Literal["de", "da", "ds"]  # data engineering | data analysis | data science
# `expert` sits above `hard`: problems built around a named correctness trap where the obvious
# solution is wrong. `hard` was saturated by a strong 35B base (Ornith-1.5 scored ~18/18 on the
# seed), leaving no headroom for a before/after delta -- `expert` restores it. See ADR-002.
Difficulty = Literal["easy", "medium", "hard", "expert"]
Mode = Literal["python", "sql"]

CATEGORY_NAMES = {"de": "data engineering", "da": "data analysis", "ds": "data science"}
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard", "expert")
CATEGORIES: tuple[Category, ...] = ("de", "da", "ds")


@dataclass(frozen=True)
class Problem:
    """One benchmark task. Keep it small enough to read in one screen."""

    id: str
    category: Category
    difficulty: Difficulty
    title: str
    prompt: str
    mode: Mode
    # Deterministic fixtures. python mode -> kwargs for the model's function.
    # sql mode -> {"tables": {name: DataFrame}}.
    make_inputs: Callable[[], dict[str, Any]]
    # Given the model's produced value (python return value, or sql result DataFrame),
    # decide pass/fail. May return bool, or (bool, reason) so failures explain themselves.
    check: Callable[[Any], bool | tuple[bool, str]]
    # The gold solution, as the exact string a perfect model would emit. Powers `dsbench-selftest`,
    # which runs every reference through the real sandbox+checker so we trust the checkers before
    # we ever trust a score. Also documents the intended answer for whoever grows the set.
    reference: str
    entrypoint: str = "solve"  # python mode: the function name the prompt asks for
    timeout: float = 25.0  # seconds; a runaway candidate is a fail, not a hung run
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class ProblemResult:
    """One (problem, attempt) outcome, recorded verbatim so a failure is debuggable later."""

    id: str
    category: Category
    difficulty: Difficulty
    passed: bool
    # ok | wrong | error | timeout | no_code | truncated | http_error
    # `truncated` = hit the token cap with no answer (runaway reasoning); an artifact, not a miss.
    status: str
    reason: str = ""
    extracted_code: str = ""
    raw_response: str = ""
    latency_s: float = 0.0
