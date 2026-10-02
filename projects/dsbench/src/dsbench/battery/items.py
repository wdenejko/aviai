"""The unit of work: one benchmark item, plus the JSONL files for items, generations and scores.

An item holds what the model sees (`messages`, plus `gen` for per-item decoding limits and tools)
and, separately, what only the scorer may see (`ref`: tests, gold answers, answer keys). Keeping
the two in different fields is what lets `generate.py` prove it sends nothing but `messages`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATES = ("base", "adapter", "base_rep", "adapter_half")
# LoRA scale per state. base_rep is a second base pass: the A/A control for harness noise.
# adapter_half is ADR-001 Gate 2 step 4: on a regression, "halve the adapter scale at load and
# re-evaluate" -- here per request, on the same server, so it is paired with base like adapter.
STATE_SCALE = {"base": 0.0, "adapter": 1.0, "base_rep": 0.0, "adapter_half": 0.5}


@dataclass
class Item:
    bench: str
    id: str
    messages: list[dict[str, Any]]
    gen: dict[str, Any] = field(default_factory=dict)  # max_tokens, stop, tools, ...
    ref: dict[str, Any] = field(default_factory=dict)  # scorer-only: never sent to the model
    meta: dict[str, Any] = field(default_factory=dict)  # strata for reporting (library, difficulty)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def append_row(fd: int, row: dict) -> None:
    """One row, one write to a file opened O_APPEND. A pass stopped by a signal (the window's time
    limit) leaves whole rows behind, so it resumes; a thinking row can run to 50 KB, which a
    buffered writer may split across several writes."""
    data = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    while data:
        data = data[os.write(fd, data):]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_items(path: str | Path) -> list[Item]:
    return [Item(**row) for row in read_jsonl(path)]


def save_items(path: str | Path, items: Iterable[Item]) -> int:
    return write_jsonl(path, (asdict(it) for it in items))


def by_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index generation/score rows by item id; a later row for the same id wins (resume retries)."""
    return {row["id"]: row for row in rows}
