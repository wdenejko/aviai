"""TAF gold-anchor calibration — audit the 2-voice consensus against AWC's official decode.

The TAF sibling of consensus/gold.py. The consensus (avwx+mivek) catches disagreement, but only
an independent AUTHORITY catches a plausible-but-wrong value the two code parsers happen to share.
AWC's bundled `<forecast>` decode is that authority (NOAA's own, machine-readable) — the TAF
analogue of the AWC METAR decode used to calibrate the METAR consensus.

`calibrate_taf` scores the consensus against the AWC anchor per (period, field) and reports
per-field accuracy + every divergence. Read the result with one caveat baked in by measurement:
AWC's decode is **SM-native**, so on our metres-dominant corpus its `visibility_m` carries
round-trip rounding noise — a lower vis agreement here indicts AWC's SM round-trip, not the
consensus. Wind / clouds / change_type / cavok agreement is the real trust signal.
"""

from __future__ import annotations

from dataclasses import dataclass

from avtext.consensus.vote_taf import PERIOD_FIELDS, consensus_taf, flatten_period
from avtext.oracles.avwx_taf import AvwxTafOracle
from avtext.oracles.awc_taf_voice import decode_awc_taf
from avtext.oracles.mivek_taf import MivekTafOracle

_AVWX, _MIVEK = AvwxTafOracle(), MivekTafOracle()


@dataclass
class TafCalibration:
    per_field: dict[str, tuple[int, int, float | None]]  # field -> (n_compared, n_match, accuracy)
    divergences: list[
        tuple[str, int, str, object, object]
    ]  # (station, period, field, consensus, gold)
    n_records: int


def calibrate_taf(taf_elements: list) -> TafCalibration:
    """Score the avwx+mivek consensus against AWC's decode over a set of <TAF> XML elements.

    Only structurally-agreeing records are audited (a structural hard case is not a field-value
    question). Both sides are flattened identically (nearest-100 m vis etc.) so the comparison is
    like-for-like; a field is compared only where both consensus and anchor have an opinion."""
    tally = {f: [0, 0] for f in PERIOD_FIELDS}
    divergences: list[tuple[str, int, str, object, object]] = []
    n = 0
    for el in taf_elements:
        raw = el.findtext("raw_text")
        if not raw:
            continue
        ra, rm = _AVWX.decode(raw), _MIVEK.decode(raw)
        if not (ra.ok and rm.ok):
            continue
        anchor = decode_awc_taf(el)
        if anchor is None:
            continue
        con = consensus_taf({"avwx": ra.forecast, "mivek": rm.forecast})
        if con.structural_dissent:
            continue
        n += 1
        agold = [flatten_period(p) for p in anchor.periods]
        for i, cper in enumerate(con.periods):
            if i >= len(agold):
                break
            for f in PERIOD_FIELDS:
                cv, gv = cper.get(f), agold[i].get(f)
                if cv is None or gv is None:
                    continue  # compare only where both have an opinion
                tally[f][0] += 1
                if cv == gv:
                    tally[f][1] += 1
                else:
                    divergences.append((con.header["station"], i, f, cv, gv))
    per_field = {f: (nn, mm, (mm / nn if nn else None)) for f, (nn, mm) in tally.items()}
    return TafCalibration(per_field, divergences, n)
