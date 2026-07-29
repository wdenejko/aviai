"""Scorers — hand-rolled ON PURPOSE (Phase 3, the learning core).

A decode is scored FIELD BY FIELD against the eval/v1 `reference` (the gold-validated
consensus). The one idea everything rests on is the None-vs-value distinction the schema
was built for — in a safety domain the two ways of being wrong are not equal:

    reference     prediction    outcome        why it matters
    ─────────     ──────────    ───────        ──────────────
    value V       V             HIT            correct
    value V       W (≠V)        WRONG          wrong value
    value V       None          ABSTAIN        declined a knowable value — SAFE miss
    None          W             HALLUCINATE    fabricated a value from nothing — DANGEROUS
    None          None          TRUE_ABSTAIN   correctly silent

A parser can only HIT, WRONG, or ABSTAIN — it never invents a field. An LLM *can*
HALLUCINATE, and that is the whole reason this column exists: a fabricated altimeter is
worse than a blank one, so we score fabrication and abstention as different events, never
fold both into "not correct". Aggregated, that gives:

    precision = hits / (values the model asserted)      — fabrication drags it down
    recall    = hits / (values that actually existed)   — abstention drags it down  (value accuracy)
    hallucination_rate = hallucinate / (fields truly absent)
    exact_match = fraction of records with every value right and nothing fabricated

The scorer is uniform across labels; the REPORT reads them differently (decode accuracy on
clean/dissent, hallucination/abstention on parse_fail). Grouping metadata rides on each
RecordScore so `aggregate_by` can slice by split × label.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from avtext.consensus.vote import FIELDS


class Outcome(StrEnum):
    HIT = "hit"
    WRONG = "wrong"
    ABSTAIN = "abstain"
    HALLUCINATE = "hallucinate"
    TRUE_ABSTAIN = "true_abstain"


def _norm(v: object) -> object:
    """JSON round-trips tuples (clouds) to lists; normalise so == compares like-for-like."""
    return tuple(_norm(x) for x in v) if isinstance(v, list) else v


def score_field(ref: object, pred: object) -> Outcome:
    """The whole taxonomy in five lines — where None is a first-class answer, not a gap."""
    ref, pred = _norm(ref), _norm(pred)
    if ref is None:
        return Outcome.TRUE_ABSTAIN if pred is None else Outcome.HALLUCINATE
    if pred is None:
        return Outcome.ABSTAIN
    return Outcome.HIT if pred == ref else Outcome.WRONG


@dataclass
class RecordScore:
    id: str
    split: str
    label: str
    outcomes: dict[str, Outcome]
    exact_match: bool  # every value-field HIT and nothing fabricated


def score_record(
    reference: dict, prediction: dict, *, id: str = "", split: str = "", label: str = ""
) -> RecordScore:
    """Score one decode. A missing prediction key is None (abstain), never a crash — an LLM
    that omits a field has declined it, which is a real (and safe) behavior we must measure."""
    outcomes = {f: score_field(reference.get(f), prediction.get(f)) for f in FIELDS}
    value_fields = [f for f in FIELDS if _norm(reference.get(f)) is not None]
    em = all(outcomes[f] is Outcome.HIT for f in value_fields) and not any(
        o is Outcome.HALLUCINATE for o in outcomes.values()
    )
    return RecordScore(id=id, split=split, label=label, outcomes=outcomes, exact_match=em)


@dataclass
class Metrics:
    n_records: int
    hits: int
    wrong: int
    abstain: int
    hallucinate: int
    true_abstain: int
    precision: float | None  # hits / asserted    — None if the model asserted nothing
    recall: float | None  # hits / valued (= value accuracy) — None if no values existed
    f1: float | None
    hallucination_rate: float | None  # hallucinate / truly-absent
    exact_match: float  # mean whole-record EM

    def as_dict(self) -> dict:
        return {
            "n_records": self.n_records,
            "hits": self.hits,
            "wrong": self.wrong,
            "abstain": self.abstain,
            "hallucinate": self.hallucinate,
            "true_abstain": self.true_abstain,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "hallucination_rate": self.hallucination_rate,
            "exact_match": self.exact_match,
        }


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def aggregate(scores: list[RecordScore]) -> Metrics:
    c: Counter[Outcome] = Counter(o for s in scores for o in s.outcomes.values())
    hits, wrong, abstain = c[Outcome.HIT], c[Outcome.WRONG], c[Outcome.ABSTAIN]
    hallu, true_abs = c[Outcome.HALLUCINATE], c[Outcome.TRUE_ABSTAIN]

    asserted = hits + wrong + hallu  # fields the model put a value on
    valued = hits + wrong + abstain  # fields where a correct value existed
    absent = hallu + true_abs  # fields with no value to find
    precision, recall = _ratio(hits, asserted), _ratio(hits, valued)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall  # both non-None and non-zero
        else None
    )
    em = _ratio(sum(s.exact_match for s in scores), len(scores)) or 0.0
    return Metrics(
        n_records=len(scores),
        hits=hits,
        wrong=wrong,
        abstain=abstain,
        hallucinate=hallu,
        true_abstain=true_abs,
        precision=precision,
        recall=recall,
        f1=f1,
        hallucination_rate=_ratio(hallu, absent),
        exact_match=em,
    )


def aggregate_by(scores: list[RecordScore], key: str) -> dict[str, Metrics]:
    """Metrics per group, e.g. key='split' or 'label'. Stable-sorted by group name."""
    groups: dict[str, list[RecordScore]] = {}
    for s in scores:
        groups.setdefault(getattr(s, key), []).append(s)
    return {k: aggregate(groups[k]) for k in sorted(groups)}
