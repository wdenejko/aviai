"""TAF scorer — period-aligned five-way scoring (Phase 6).

The TAF sibling of harness/score.py. A TAF reference is nested (header + ordered periods), so we
score it in two parts and reuse the METAR taxonomy verbatim (`score_field`: HIT / WRONG / ABSTAIN
/ HALLUCINATE / TRUE_ABSTAIN — the None-vs-value distinction that makes fabrication and abstention
different events):

  • header fields (station, issue/validity times, AMD/COR/NIL/CNL), keyed `h.<field>`;
  • period fields, aligned BY SEQUENCE (period i of prediction vs period i of reference — the same
    alignment the consensus uses), keyed `p<i>.<field>`.

Period-count mismatch falls out of the taxonomy for free: a period the model OMITS leaves every
reference field with prediction None -> ABSTAIN (a safe miss); a period the model INVENTS beyond
the reference has ref None, pred value -> HALLUCINATE (the dangerous kind). So "made up an extra
TEMPO group" is scored as fabrication and "dropped a group" as abstention, exactly as they should
be. Whole-forecast EM requires every reference value HIT and nothing hallucinated anywhere —
deliberately strict, because getting the change-group structure right is the point of TAF.

RecordScore/aggregate are reused from score.py, so the runner's reporting and stats work unchanged.
"""

from __future__ import annotations

from avtext.harness.prompt_taf import HEADER_FIELDS, PERIOD_ALL_FIELDS
from avtext.harness.score import Outcome, RecordScore, score_field

_PRESENT = (Outcome.HIT, Outcome.WRONG, Outcome.ABSTAIN)  # a value existed in the reference


def score_taf_record(
    reference: dict, prediction: dict | None, *, id: str = "", split: str = "", label: str = ""
) -> RecordScore:
    """Score one nested TAF decode against its consensus reference. A None prediction (unparseable
    output) scores as abstention everywhere — never a crash."""
    pred = prediction or {}
    ref_h = reference.get("header") or {}
    pred_h = pred.get("header") or {}
    outcomes: dict[str, Outcome] = {
        f"h.{f}": score_field(ref_h.get(f), pred_h.get(f)) for f in HEADER_FIELDS
    }

    ref_p = reference.get("periods") or []
    pred_p = pred.get("periods") or []
    for i in range(max(len(ref_p), len(pred_p))):
        rp = ref_p[i] if i < len(ref_p) else {}
        pp = pred_p[i] if i < len(pred_p) else {}
        for f in PERIOD_ALL_FIELDS:
            outcomes[f"p{i}.{f}"] = score_field(rp.get(f), pp.get(f))

    # EM: every field that HAD a value is HIT, and nothing was fabricated anywhere.
    em = all(o is Outcome.HIT for o in outcomes.values() if o in _PRESENT) and not any(
        o is Outcome.HALLUCINATE for o in outcomes.values()
    )
    return RecordScore(id=id, split=split, label=label, outcomes=outcomes, exact_match=em)


def period_count_match(reference: dict, prediction: dict | None) -> bool:
    """Did the model produce the right NUMBER of change groups? A TAF-specific structural signal
    reported alongside the field metrics (a wrong count guarantees EM=0 but is worth seeing raw)."""
    ref_n = len(reference.get("periods") or [])
    pred_n = len((prediction or {}).get("periods") or [])
    return ref_n == pred_n
