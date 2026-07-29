"""Oracle interface + shared helpers (Phase 2, Skill B).

Three independent parsers map raw METAR text into the canonical MetarObservation.
This module holds the common result type plus the bits not worth parsing three
times: the fixed-position lexical header (type / station / DDHHMMZ / AUTO / COR /
CAVOK) and unit conversions to the schema's canonical units. Independent parsing is
reserved for the meteorological fields (wind, visibility, clouds, temp/dewpoint,
altimeter) — that's where parsers genuinely differ and where cross-oracle
disagreement is a real signal.

Why parse the header here, not per-parser: it's fixed-position lexical, and the
parsers are quirky about it — e.g. python-metar defaults `mod='AUTO'` even when the
report has no AUTO, and avwx leaves `type` unset. AUTO/COR are literal tokens, so
we read them straight from the raw and keep the adapters focused on the meaning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from avtext.schema import AltimeterSource, MetarObservation, ReportType

# Optional leading report type, then an optional COR/AMD modifier, then station +
# DDHHMMZ. IEM sometimes emits the correction flag first ("COR EGCC 271850Z ...").
_HEADER_RE = re.compile(
    r"^(?:(METAR|SPECI)\s+)?(?:(?:COR|AMD)\s+)?([A-Z][A-Z0-9]{2,3})\s+(\d{2})(\d{2})(\d{2})Z"
)


@dataclass
class OracleResult:
    """Uniform result across parsers with different failure conventions
    (python-metar/mivek raise; avwx returns nulls)."""

    oracle: str
    ok: bool
    obs: MetarObservation | None = None
    error: str | None = None


def parse_header(raw: str) -> tuple[ReportType, str, int, int, int]:
    """(report_type, station, day, hour, minute) from the lexical header."""
    m = _HEADER_RE.match(raw.strip())
    if not m:
        raise ValueError(f"unparseable METAR header: {raw[:40]!r}")
    rtype, station, dd, hh, mm = m.groups()
    return (ReportType(rtype) if rtype else ReportType.METAR, station, int(dd), int(hh), int(mm))


def detect_modifiers(raw: str) -> tuple[bool, bool]:
    """(automated, corrected) from literal AUTO / COR tokens."""
    toks = raw.split()
    return "AUTO" in toks, "COR" in toks


def is_cavok(raw: str) -> bool:
    # Only the main body counts: a trailing "BECMG CAVOK" forecasts CAVOK, it does
    # not describe current conditions. Truncate at the first trend/remark keyword.
    body = re.split(r"\b(?:BECMG|TEMPO|NOSIG|RMK|PROB\d\d|FM\d)\b", raw, maxsplit=1)[0]
    return "CAVOK" in body.split()


def altimeter_source(raw: str) -> AltimeterSource | None:
    for t in raw.split():
        if re.fullmatch(r"Q\d{4}", t):
            return AltimeterSource.Q
        if re.fullmatch(r"A\d{4}", t):
            return AltimeterSource.A
    return None


# -- unit conversions -> the schema's canonical units --
def mps_to_kt(v: float | None) -> float | None:
    return None if v is None else v * 1.94384


def sm_to_m(v: float | None) -> float | None:
    return None if v is None else v * 1609.344


def inhg_to_hpa(v: float | None) -> float | None:
    return None if v is None else v * 33.8639


class Oracle:
    """Base adapter. Subclasses implement `_decode(raw) -> MetarObservation | None`;
    `decode` wraps it in a uniform OracleResult, normalizing heterogeneous failures."""

    name: str = "oracle"

    def _decode(self, raw: str) -> MetarObservation | None:
        raise NotImplementedError

    def decode(self, raw: str) -> OracleResult:
        try:
            obs = self._decode(raw)
        except Exception as e:  # noqa: BLE001 — normalize every parser's failure mode
            return OracleResult(self.name, ok=False, error=f"{type(e).__name__}: {e}")
        if obs is None:
            return OracleResult(self.name, ok=False, error="parser produced no observation")
        return OracleResult(self.name, ok=True, obs=obs)
