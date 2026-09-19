"""The contract for one generated SFT row -- the sftgen analogue of ADR-003's `AgentProblem`.

A row is a small chat conversation (system / user / assistant) plus the metadata that makes it
auditable: WHERE the label came from (`provenance`) and HOW we know it is correct (`verification`).
The trainer consumes only the `turns`; everything else exists so the decontamination gate and the
licence audit (ADR-001 Gate 3) can run over the mixture without re-deriving anything.

Why a neutral turn/row shape rather than emitting Qwen ChatML directly: the same rows must render
into the corrected Qwen3.5/3.6 template (assistant-only loss, thinking on/off) AND be scannable by
`decontaminate.py`. Rendering is a separate concern (`render.py`); this module is the source of
truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Turn:
    """One chat turn. `loss=True` marks turns the trainer computes loss on (assistant-only)."""

    role: str  # system | user | assistant
    content: str
    loss: bool = False
    # If True and this is an assistant turn, the renderer wraps a real reasoning trace in <think>.
    # If False, the renderer emits an EMPTY <think></think> (ADR-001 B.4 non-thinking convention).
    thinking: str | None = None


@dataclass(frozen=True)
class Provenance:
    """WHERE the row came from -- the licence-audit record (ADR-001 B.4 / ADR-004)."""

    generator: str  # e.g. "dialect_conventions"
    method: str  # "own-generated" | "teacher-distilled" | "teacher-in-sandbox"
    teacher: str | None = None  # model id if a teacher authored any text; None if teacher-free
    licence: str = "Apache-2.0"  # of the redistributable row (own-generated => Apache-2.0)
    seed: int | None = None  # the synthetic-data seed, so the row is reproducible


@dataclass(frozen=True)
class Verification:
    """HOW we know the label is correct -- the execution proof (ADR-004 'kept only if agree')."""

    engine: str  # the dialect engine the SQL was executed against (duckdb | clickhouse | ...)
    truth: Any  # the independent pandas re-derivation
    engine_result: Any  # what the SQL returned when executed
    agrees: bool  # truth == engine_result (only agreeing rows are emitted)


@dataclass(frozen=True)
class SFTRow:
    """One execution-verified training example."""

    id: str  # stable, e.g. "A-weekday-numbering-clickhouse-retail-0007"
    target: str  # "A" | "B" | "C"
    family: str  # convention/trap family, e.g. "weekday-numbering"
    dialect: str  # the SQL dialect the row teaches (duckdb | clickhouse | postgres | mysql)
    turns: tuple[Turn, ...]
    provenance: Provenance
    verification: Verification
    tags: tuple[str, ...] = field(default_factory=tuple)

    def text_blob(self) -> str:
        """All natural-language + SQL content, for the decontamination scan (ADR-004)."""
        parts = [t.content for t in self.turns]
        parts += [t.thinking for t in self.turns if t.thinking]
        return "\n".join(parts)


def row_to_dict(row: SFTRow) -> dict[str, Any]:
    """Plain-dict form for the raw JSONL (the source-of-truth artifact)."""
    return asdict(row)


def row_from_dict(d: dict[str, Any]) -> SFTRow:
    return SFTRow(
        id=d["id"], target=d["target"], family=d["family"], dialect=d["dialect"],
        turns=tuple(Turn(**t) for t in d["turns"]),
        provenance=Provenance(**d["provenance"]),
        verification=Verification(**d["verification"]),
        tags=tuple(d.get("tags", ())),
    )
