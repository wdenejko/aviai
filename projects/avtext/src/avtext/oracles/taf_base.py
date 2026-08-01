"""TAF oracle interface + shared lexical/mapping helpers (Phase 6).

The TAF sibling of oracles/base.py. Same division of labour: the **header** (report modifiers +
station + issue DDHHMMZ + validity DDHH/DDHH + NIL/CNL) is fixed-position lexical, so we read it
straight from the raw here — once — rather than trusting each parser's header handling. The
adapters then only map the meteorological **change groups** into ForecastPeriod, which is where
avwx and mivek genuinely differ and where cross-voice disagreement is real signal.

Two TAF-specific mapping helpers live here because every adapter needs them identically:
  • change-type normalisation (each library names FM/BECMG/TEMPO/PROB… differently), and
  • the P6SM / '≥' visibility convention → (visibility_m, visibility_plus).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from avtext.schema import ChangeType, CloudCover, SkyClear, TafForecast

# TAF header: optional 'TAF', optional AMD/COR, station, issue DDHHMMZ, optional validity DDHH/DDHH.
# The validity is absent on a NIL TAF; CNL/NIL are detected from the token stream below.
_TAF_HEADER = re.compile(
    r"^(?:TAF\s+)?(?:(AMD|COR)\s+)?"
    r"([A-Z][A-Z0-9]{2,3})\s+"
    r"(\d{2})(\d{2})(\d{2})Z"
    r"(?:\s+(\d{2})(\d{2})/(\d{2})(\d{2}))?"
)

# Cover/sky maps reused by every adapter (identical to METAR — a forecast layer is a layer).
COVER = {"FEW": CloudCover.FEW, "SCT": CloudCover.SCT, "BKN": CloudCover.BKN, "OVC": CloudCover.OVC}
SKY_CLEAR = {"SKC": SkyClear.SKC, "CLR": SkyClear.CLR, "NSC": SkyClear.NSC, "NCD": SkyClear.NCD}


@dataclass
class TafResult:
    """Uniform result across TAF parsers with different failure conventions (mivek raises;
    avwx returns nulls) — the TAF analogue of OracleResult."""

    oracle: str
    ok: bool
    forecast: TafForecast | None = None
    error: str | None = None


def parse_taf_header(raw: str) -> dict:
    """Decode the lexical TAF header into the TafForecast header fields (a dict to spread in).

    Handles the routine form `[TAF] [AMD|COR] <stn> DDHHMMZ DDHH/DDHH`, plus NIL (no validity)
    and CNL (a cancelled forecast). Raises on an unparseable header so a truly malformed record
    is a parser failure, not a silently-empty forecast."""
    m = _TAF_HEADER.match(raw.strip())
    if not m:
        raise ValueError(f"unparseable TAF header: {raw[:48]!r}")
    amdcor, station, dd, hh, mm, vfd, vfh, vtd, vth = m.groups()
    toks = raw.split()
    return {
        "station": station,
        "issue_day": int(dd),
        "issue_hour": int(hh),
        "issue_minute": int(mm),
        "valid_from_day": int(vfd) if vfd else None,
        "valid_from_hour": int(vfh) if vfh else None,
        "valid_to_day": int(vtd) if vtd else None,
        "valid_to_hour": int(vth) if vth else None,
        "amended": amdcor == "AMD",
        "corrected": amdcor == "COR",
        "is_nil": "NIL" in toks,
        "is_cancelled": "CNL" in toks,
    }


def change_type(kind: str, *, prob: int | None = None, is_tempo: bool = False) -> ChangeType:
    """Normalise a parser's group label (+ optional probability / tempo flag) to our ChangeType.

    `kind` is the library's own token ('FROM'/'FM', 'BECMG', 'TEMPO', 'INITIAL'/'MAIN', 'PROB…').
    A PROB group can combine with TEMPO (`PROB30 TEMPO`), which the callers signal via is_tempo."""
    k = (kind or "").upper()
    if prob in (30, 40):
        return ChangeType(f"PROB{prob} TEMPO") if is_tempo else ChangeType(f"PROB{prob}")
    if k in ("FM", "FROM"):
        return ChangeType.FM
    if k == "BECMG":
        return ChangeType.BECMG
    if k == "TEMPO":
        return ChangeType.TEMPO
    return ChangeType.INITIAL  # MAIN / OBSERVATION / the base group


def visibility_from_sm(value: float | None, repr_str: str | None) -> tuple[int | None, bool]:
    """Map a statute-mile forecast visibility to (metres, plus).

    US TAFs use SM; `P6SM` ('more than 6 SM') is the US analogue of METAR 9999 and carries a
    'plus' flag so '>6SM' never collapses onto 'exactly 6SM'. A parser may signal P6SM either by
    a 'P6'-style repr with a null value (avwx) or by the literal in the repr; we handle both."""
    r = (repr_str or "").upper()
    plus = r.startswith("P") or ">" in r
    if value is None:
        # avwx nulls the value on P6SM but keeps repr 'P6' — recover the 6 SM floor.
        return (9656, True) if plus else (None, False)  # 6 SM = 9656 m
    from avtext.oracles.base import sm_to_m

    return int(round(sm_to_m(value))), plus


class TafOracle:
    """Base adapter — subclasses implement `_decode(raw) -> TafForecast | None`; `decode` wraps
    it in a uniform TafResult, normalising each library's failure mode (mirror of Oracle)."""

    name: str = "taf_oracle"

    def _decode(self, raw: str) -> TafForecast | None:
        raise NotImplementedError

    def decode(self, raw: str) -> TafResult:
        try:
            fc = self._decode(raw)
        except Exception as e:  # noqa: BLE001 — normalise every parser's failure mode
            return TafResult(self.name, ok=False, error=f"{type(e).__name__}: {e}")
        if fc is None:
            return TafResult(self.name, ok=False, error="parser produced no forecast")
        return TafResult(self.name, ok=True, forecast=fc)
