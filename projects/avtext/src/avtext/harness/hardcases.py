"""Hard-case queue — the eval harness's first-class view of the messy tail (Phase 3).

`classify` runs one raw report through the whole trust ladder and returns a single
structured verdict: which tier (if any) flagged it, plus the consensus field values
that serve as the scoring reference. It is the labeling function the eval set is built
from — and the reason that set can be *stratified on difficulty* instead of sampled
blind. ADR-004's thesis is that the interesting signal lives in this tail (clean
decodes are already a solved problem for the parsers), so the harness treats the tail
as a named artifact, never as noise to discard.

Relationship to `consensus.mutants.detect`: same ladder, but detect() returns only
the channel name — it is the audit's yes/no probe. classify() is the richer superset
the harness needs: it keeps every signal *and* the reference answer. They are kept
separate to preserve layering — harness sits above consensus, so consensus must never
import harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from avtext.consensus.vote import consensus
from avtext.oracles import ORACLES
from avtext.quality import check_invariants

# Headline buckets. A single report can trip several signals at once; `label` records
# the HARDEST one (a parse failure is a deeper problem than a mere vote tie), while the
# verdict below keeps the full signal set so the sampler can slice on any of them.
CLEAN = "clean"  # every voice parsed, agreed unanimously, invariants pass
PARSE_FAIL = "parse_fail"  # >=1 oracle raised / returned null
INVARIANT = "invariant"  # a decoded reading is physically impossible
NO_MAJORITY = "no_majority"  # voices tie -> NO trustworthy reference (human/gold queue)
DISSENT = "dissent"  # a majority resolved it, but not unanimously (hard, yet answerable)

# Priority order, hardest first — the order the headline label is chosen in, and the
# stable key order for any distribution table. DISSENT sits just above CLEAN: it is the
# mildest "not trivial" signal, but (unlike NO_MAJORITY) it keeps a trustworthy
# reference, which makes it the richest auto-scorable difficulty stratum (ADR-004).
LABELS = (PARSE_FAIL, INVARIANT, NO_MAJORITY, DISSENT, CLEAN)


@dataclass
class CaseVerdict:
    raw: str
    label: str  # one of LABELS — the headline bucket (hardest signal wins)
    n_ok: int  # oracles that decoded, out of len(ORACLES)
    failed: list[str]  # oracle names that raised / returned null
    invariant_hits: list[str]  # invariant violations, unioned across decoders
    disagreements: list[str]  # fields with no majority (a tie for the top) -> no reference
    dissent: list[str]  # fields a majority resolved but the voices didn't all agree on
    reference: dict[str, Any]  # consensus field values -> the scoring target


def classify(raw: str) -> CaseVerdict:
    """Run `raw` through oracles -> invariants -> consensus and label it once."""
    results = [o.decode(raw) for o in ORACLES]
    ok = [r for r in results if r.ok]
    failed = [r.oracle for r in results if not r.ok]
    obs = [r.obs for r in ok]

    # Invariants are spec rules and parser-independent, so we take the UNION of what
    # every successful decode violates: if even one plausible reading is physically
    # impossible, the record is worth flagging. Dedup keeps the message list readable
    # when two parsers report the same violation.
    seen: set[str] = set()
    invariant_hits: list[str] = []
    for o in obs:
        for msg in check_invariants(o):
            if msg not in seen:
                seen.add(msg)
                invariant_hits.append(msg)

    # Consensus over whoever parsed. For a parse_fail this reference is partial (only
    # the surviving voices vote) — that's intentional: the scorer decides later whether
    # a partial/absent reference means "abstain" is the correct model behavior.
    con = consensus(obs) if obs else None
    disagreements = con.disagreements if con else []

    # Dissent = a field the majority resolved but not unanimously (agreement < 1.0,
    # yet not a tie). con.agreement[f] is top_count / #voices-with-an-opinion, so 1.0
    # means unanimous and a lone single voice can't dissent. This is the difficulty
    # axis with a *kept* reference — the scope probe's ~3.4% lives here.
    dissent = (
        [
            f
            for f, agr in con.agreement.items()
            if agr is not None and agr < 1.0 and f not in disagreements
        ]
        if con
        else []
    )
    reference = con.values if con else {}

    # Hardest signal wins the headline bucket.
    if failed:
        label = PARSE_FAIL
    elif invariant_hits:
        label = INVARIANT
    elif disagreements:
        label = NO_MAJORITY
    elif dissent:
        label = DISSENT
    else:
        label = CLEAN

    return CaseVerdict(
        raw=raw,
        label=label,
        n_ok=len(ok),
        failed=failed,
        invariant_hits=invariant_hits,
        disagreements=disagreements,
        dissent=dissent,
        reference=reference,
    )


def summarize(verdicts: list[CaseVerdict]) -> dict[str, int]:
    """Count verdicts per headline label (stable LABELS order, zeros included)."""
    counts = {lbl: 0 for lbl in LABELS}
    for v in verdicts:
        counts[v.label] += 1
    return counts
