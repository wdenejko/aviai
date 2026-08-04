"""NOTAM scorer — row-aligned five-way field scoring (Phase 7).

Reuses the METAR/TAF taxonomy (`score_field`: HIT/WRONG/ABSTAIN/HALLUCINATE/TRUE_ABSTAIN) on the
per-category fields (`NOTAM_FIELDS[category]`). The one NOTAM-specific choice is ROW ALIGNMENT: a
NOTAM affects several subjects (runways, taxiways…) whose rows have **no inherent order**, so unlike
TAF (positional) we align rows by **greedy best-match** — each reference row is paired with the
still-unused prediction row that agrees on the most non-null fields. This is order-invariant, so
"same content, different order" scores correct, while:

  • a reference row with no prediction match  → every field ABSTAIN (a safe miss),
  • an extra invented prediction row          → every stated field HALLUCINATE (the dangerous kind).

Whole-NOTAM EM = every reference value HIT and nothing hallucinated. RecordScore/aggregate are
reused from score.py, so the runner's reporting and stats work unchanged.
"""

from __future__ import annotations

from avtext.harness.score import Outcome, RecordScore, score_field
from avtext.schema.notam import NOTAM_FIELDS

_PRESENT = (Outcome.HIT, Outcome.WRONG, Outcome.ABSTAIN)


def _match_score(ref_row: dict, pred_row: dict) -> int:
    """How many non-null reference fields the prediction row gets right (the pairing key)."""
    return sum(1 for f, v in ref_row.items() if v is not None and pred_row.get(f) == v)


def align_rows(ref_rows: list[dict], pred_rows: list[dict]) -> list[tuple[dict, dict]]:
    """Greedy best-match pairing (order-invariant). Returns (ref_row, pred_row) pairs; an unmatched
    side is paired with {} so the field scorer reads it as all-abstain / all-hallucinate."""
    remaining = list(range(len(pred_rows)))
    pairs: list[tuple[dict, dict]] = []
    for rr in ref_rows:
        best_j, best_s = None, -1
        for j in remaining:
            s = _match_score(rr, pred_rows[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None:
            remaining.remove(best_j)
            pairs.append((rr, pred_rows[best_j]))
        else:
            pairs.append((rr, {}))  # no prediction row left → abstain on this reference row
    for j in remaining:
        pairs.append(({}, pred_rows[j]))  # extra invented prediction row → hallucinate
    return pairs


def score_notam_record(
    reference: dict, prediction: dict | None, *, id: str = "", split: str = "", label: str = ""
) -> RecordScore:
    """Score one NOTAM extraction against its Knots gold. `reference`/`prediction` are
    {category, rows:[...]}; a None prediction scores as abstention everywhere."""
    pred = prediction or {}
    category = reference.get("category", "")
    fields = NOTAM_FIELDS.get(category, ())
    ref_rows = reference.get("rows") or []
    pred_rows = pred.get("rows") or []

    outcomes: dict[str, Outcome] = {}
    for i, (rr, pr) in enumerate(align_rows(ref_rows, pred_rows)):
        for f in fields:
            outcomes[f"r{i}.{f}"] = score_field(rr.get(f), pr.get(f))

    em = all(o is Outcome.HIT for o in outcomes.values() if o in _PRESENT) and not any(
        o is Outcome.HALLUCINATE for o in outcomes.values()
    )
    return RecordScore(id=id, split=split, label=label, outcomes=outcomes, exact_match=em)


def row_count_match(reference: dict, prediction: dict | None) -> bool:
    """Did the model get the right NUMBER of affected subjects? A NOTAM structural signal."""
    ref_n = len(reference.get("rows") or [])
    pred_n = len((prediction or {}).get("rows") or [])
    return ref_n == pred_n
