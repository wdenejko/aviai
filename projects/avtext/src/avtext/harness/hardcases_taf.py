"""TAF hard-case classifier — label a raw TAF + attach its consensus reference (Phase 6).

The TAF sibling of harness/hardcases.py, reusing the same three labels so the eval reads the
same way across products:

    CLEAN       both voices decode AND the period-aligned consensus fully agrees — a trustworthy
                whole-forecast reference to score decode accuracy + exact-match against.
    DISSENT     both voices decode but disagree on some (period, field), OR only one voice decoded
                (an unvalidated single-voice reference) — the "needs a careful look" bucket.
    PARSE_FAIL  neither voice decoded — no reference; tests abstention vs fabrication.

The reference is the 2-voice consensus (avwx+mivek); AWC is the aggregate gold ANCHOR
(consensus/gold_taf.py), not a per-record voice — see its docstring for why (SM-native).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from avtext.consensus.vote_taf import consensus_taf
from avtext.harness.hardcases import CLEAN, DISSENT, PARSE_FAIL
from avtext.oracles.avwx_taf import AvwxTafOracle
from avtext.oracles.mivek_taf import MivekTafOracle

_AVWX, _MIVEK = AvwxTafOracle(), MivekTafOracle()


@dataclass
class TafCase:
    label: str
    reference: dict  # {"header": {...}, "periods": [ {field: value, ...}, ... ]}
    dissent: list[str] = field(default_factory=list)  # "p{i}.{field}" with no clear majority
    failed: list[str] = field(default_factory=list)  # voices that could not decode


def classify_taf(raw: str) -> TafCase:
    """Decode with both voices, build the consensus reference, and label the record."""
    ra, rm = _AVWX.decode(raw), _MIVEK.decode(raw)
    ok = {n: r.forecast for n, r in (("avwx", ra), ("mivek", rm)) if r.ok and r.forecast}
    failed = [n for n, r in (("avwx", ra), ("mivek", rm)) if not r.ok]

    if not ok:  # neither voice decoded -> abstention test, empty reference
        return TafCase(PARSE_FAIL, {"header": {}, "periods": []}, [], failed)

    con = consensus_taf(ok)
    reference = {"header": con.header, "periods": con.periods}
    if len(ok) < 2:  # single-voice decode: unvalidated -> not clean
        return TafCase(PARSE_FAIL, reference, con.disagreements, failed)
    label = DISSENT if (con.structural_dissent or con.disagreements) else CLEAN
    return TafCase(label, reference, con.disagreements, failed)
