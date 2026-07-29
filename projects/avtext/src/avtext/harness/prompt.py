"""Prompt template + output parser (Phase 3 runner).

Both are model-agnostic and VERSIONED — `PROMPT_ID` goes in every run report, because a
prompt change is as much a cause of a score change as a model change, and a result you
can't attribute to a prompt is not reproducible.

The single most important line in the template is "use null; do NOT guess." We are
measuring hallucination vs abstention, so the model must be explicitly given permission to
abstain — otherwise a fabrication might just be it obeying an implied "always answer". With
the instruction present, a fabricated value is the model's own failure, which is what the
hallucination_rate is meant to capture.
"""

from __future__ import annotations

import json

from avtext.consensus.vote import FIELDS

PROMPT_ID = "decode-json-v1"

_TEMPLATE = """\
You are an expert aviation weather decoder. Decode the METAR below into a single JSON object.

Rules:
- Output ONLY the JSON object. No prose, no explanation, no markdown fences.
- Use exactly these keys, in canonical units:
    report_type    "METAR" or "SPECI"
    automated      true if automated (contains AUTO), else false
    wind_dir       wind direction in degrees (integer), or null
    wind_speed     sustained wind in KNOTS (integer), or null
    wind_gust      gust in KNOTS (integer), or null if there is no gust
    visibility_m   prevailing visibility in METRES (integer), or null
    temperature_c  temperature in whole degrees Celsius (integer), or null
    dewpoint_c     dewpoint in whole degrees Celsius (integer), or null
    altimeter_hpa  altimeter / QNH in HECTOPASCALS (integer), or null
    clouds         list of [cover, base_feet], e.g. [["BKN", 900]]; [] if none reported
    cavok          true if the report contains CAVOK, else false
- Convert units where needed (m/s->knots, inHg->hPa, statute miles->metres).
- If a value is absent or you cannot determine it, use null. Do NOT guess.

METAR: {raw}

JSON:"""


def format_prompt(raw: str) -> str:
    return _TEMPLATE.format(raw=raw)


def _extract_json(text: str) -> dict | None:
    """Pull the first balanced {...} object out of model text (which may wrap it in prose or
    ```json fences). Returns the parsed dict, or None if nothing valid is found — a None here
    means 'no usable output', which the scorer treats as abstention on every field."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# Numeric fields the reference (vote.flatten) stores as rounded INTs — coerce the model's
# value into the same type/space so "5" (str), 5.0 (float) and 5 (int) all compare equal to
# the reference's 5. Without this a right-but-differently-typed answer scores WRONG.
_INT_FIELDS = (
    "temperature_c", "dewpoint_c", "altimeter_hpa", "wind_dir", "wind_speed", "wind_gust",
)
_BOOL_FIELDS = ("automated", "cavok")


def _canon(field: str, v: object) -> object:
    """Coerce a model field into the SAME type/space the consensus reference uses
    (vote.flatten), so scoring measures value-correctness, not formatting. Every field the
    reference canonicalises must be matched here — visibility bucketed to 100 m, numerics
    rounded to int, report_type upper-cased, bools normalised, clouds as (cover, base) tuples.
    An un-canonicalisable value is treated as absent (abstain), never a crash."""
    if v is None:
        return None
    try:
        if field == "visibility_m":
            return round(float(v) / 100) * 100  # 100 m buckets, matching flatten
        if field in _INT_FIELDS:
            return round(float(v))  # handles "16", 16.0, 16 -> 16 (int in the reference)
        if field == "report_type":
            return str(v).upper()  # "metar" -> "METAR" (reference is the enum value)
        if field in _BOOL_FIELDS:
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in ("true", "yes", "1")  # bool("false") is True — don't
            return bool(v)
        if field == "clouds":
            return tuple((str(c[0]).upper(), c[1]) for c in v)  # cover upper; list->tuple for ==
    except Exception:
        # Model output is untrusted — it may emit clouds as dicts, numbers as strings, nested
        # shapes, anything. A field we can't canonicalise is scored absent (abstain), which is
        # correct: no usable value was given in the agreed form. (A bare KeyError from
        # dict-shaped clouds used to bubble up and mark the whole record invalid.)
        return None
    return v


def parse_prediction(text: str) -> dict | None:
    """Model text -> canonical field dict (the predict() contract), or None if unparseable."""
    obj = _extract_json(text)
    if obj is None:
        return None
    return {f: _canon(f, obj.get(f)) for f in FIELDS}
