"""TAF prompt template + output parser (Phase 6 runner).

The TAF sibling of harness/prompt.py. A TAF decode is NESTED — a header plus an ordered list of
change-group periods — so the model emits `{"header": {...}, "periods": [ {...}, ... ]}`, exactly
the shape of the frozen eval's `reference` (target == reference == output, so scoring compares
like-for-like with no mapping). Same load-bearing instruction as METAR — "use null; do NOT guess",
plus "state only what a change group changes" — so abstention vs hallucination stays measurable and
the model reproduces the raw's sparsity instead of carrying values forward.

`parse_prediction_taf` canonicalises every field into the same type/space the consensus reference
uses (vote_taf.flatten_period + the header dump): ints rounded, visibility bucketed to 100 m,
clouds as [cover, base] pairs, bools normalised — so a right-but-differently-typed answer scores
HIT, not WRONG. Untrusted model output that can't be canonicalised is treated as absent (abstain),
never a crash.
"""

from __future__ import annotations

from avtext.consensus.vote_taf import PERIOD_FIELDS, TIMING_FIELDS
from avtext.harness.prompt import _extract_json  # reuse the balanced-brace JSON extractor

PROMPT_ID_TAF = "decode-taf-json-v1"

HEADER_FIELDS = (
    "station",
    "issue_day",
    "issue_hour",
    "issue_minute",
    "valid_from_day",
    "valid_from_hour",
    "valid_to_day",
    "valid_to_hour",
    "amended",
    "corrected",
    "is_nil",
    "is_cancelled",
)
PERIOD_ALL_FIELDS = PERIOD_FIELDS + TIMING_FIELDS  # votable fields + timing

_TEMPLATE = """\
You are an expert aviation weather decoder. Decode the TAF below into a single JSON object.

Output ONLY the JSON object (no prose, no markdown fences), with exactly this shape:
{{"header": {{...}}, "periods": [ {{...}}, ... ]}}

header keys:
    station         ICAO id (string)
    issue_day, issue_hour, issue_minute   from the DDHHMMZ issue time (integers)
    valid_from_day, valid_from_hour, valid_to_day, valid_to_hour   from DDHH/DDHH (ints|null)
    amended, corrected, is_nil, is_cancelled   booleans (AMD / COR / NIL / CNL)

Each period (the initial forecast, then one per FM/BECMG/TEMPO/PROB group) has keys:
    change_type   one of "INITIAL","FM","BECMG","TEMPO","PROB30","PROB40",
                  "PROB30 TEMPO","PROB40 TEMPO"
    from_day, from_hour, from_minute   when it starts (from_minute 0 unless an FM gives minutes)
    to_day, to_hour   window end for BECMG/TEMPO/PROB; null for INITIAL/FM
    probability     30 or 40 for PROB groups, else null
    wind_dir        degrees (int), null if calm/variable
    wind_speed      KNOTS (int);   wind_gust   KNOTS (int) or null
    visibility_m    prevailing visibility in METRES (int), or null
    visibility_plus true for P6SM / 'or greater'
    cavok           true if the group is CAVOK, else false
    clouds          list of [cover, base_feet], e.g. [["BKN", 900]]; [] if none stated
    sky_clear       "SKC"/"NSC"/"CLR"/"NCD" if stated, else null
    vertical_visibility_ft   VV height in feet (e.g. VV001 -> 100), or null

Rules:
- Convert units (m/s->knots, statute miles->metres). P6SM -> visibility_m 9656 + visibility_plus.
- A change group states only what CHANGES: for anything it does not mention, use null
  (do NOT carry values forward).
- If a value is absent or you cannot determine it, use null. Do NOT guess.

TAF: {raw}

JSON:"""


def format_prompt_taf(raw: str) -> str:
    return _TEMPLATE.format(raw=raw)


_INT_FIELDS = frozenset(
    {
        "issue_day", "issue_hour", "issue_minute", "valid_from_day", "valid_from_hour",
        "valid_to_day", "valid_to_hour", "probability", "wind_dir", "wind_speed", "wind_gust",
        "vertical_visibility_ft", "from_day", "from_hour", "from_minute", "to_day", "to_hour",
    }
)  # fmt: skip
_BOOL_FIELDS = frozenset(
    {"amended", "corrected", "is_nil", "is_cancelled", "cavok", "visibility_plus"}
)


def _canon(field: str, v: object) -> object:
    """Coerce a model field into the reference's type/space (see vote_taf.flatten_period)."""
    if v is None:
        return None
    try:
        if field == "visibility_m":
            return round(float(v) / 100) * 100  # 100 m buckets, matching flatten_period
        if field in _INT_FIELDS:
            return round(float(v))
        if field in _BOOL_FIELDS:
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in ("true", "yes", "1")
            return bool(v)
        if field in ("change_type", "sky_clear", "station"):
            return str(v).upper() if field != "change_type" else str(v)
        if field == "clouds":
            return [[str(c[0]).upper(), c[1]] for c in v]  # scored via score._norm (list==tuple)
    except Exception:
        return None  # untrusted output: an un-canonicalisable field is scored absent
    return v


def parse_prediction_taf(text: str) -> dict | None:
    """Model text -> {'header': {...}, 'periods': [...]} canonical form, or None if unparseable."""
    obj = _extract_json(text)
    if obj is None or not isinstance(obj, dict):
        return None
    raw_h = obj.get("header") if isinstance(obj.get("header"), dict) else {}
    header = {f: _canon(f, raw_h.get(f)) for f in HEADER_FIELDS}
    periods = []
    for p in obj.get("periods") or []:
        if not isinstance(p, dict):
            continue
        periods.append({f: _canon(f, p.get(f)) for f in PERIOD_ALL_FIELDS})
    return {"header": header, "periods": periods}
