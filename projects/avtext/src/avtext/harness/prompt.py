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


def _canon(field: str, v: object) -> object:
    """Apply the SAME canonicalisation the consensus reference used (vote.flatten), so the
    model's output is compared in the reference's space — e.g. visibility bucketed to 100 m,
    temps/altimeter rounded to integers. Otherwise the model is dinged for our bucketing."""
    if v is None:
        return None
    try:
        if field == "visibility_m":
            return round(float(v) / 100) * 100
        if field in ("temperature_c", "dewpoint_c", "altimeter_hpa"):
            return round(float(v))
        if field == "clouds":
            return tuple((c[0], c[1]) for c in v)  # list[list] -> tuple[tuple] for ==
    except (TypeError, ValueError, IndexError):
        return None  # malformed value for this field -> treat as absent, never crash
    return v


def parse_prediction(text: str) -> dict | None:
    """Model text -> canonical field dict (the predict() contract), or None if unparseable."""
    obj = _extract_json(text)
    if obj is None:
        return None
    return {f: _canon(f, obj.get(f)) for f in FIELDS}
