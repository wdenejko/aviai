"""NOTAM prompt template + output parser (Phase 7 runner).

Category-aware: the model is told which category to extract and exactly which fields that category
yields (from `NOTAM_FIELDS`), and returns `{"rows": [ {field: value}, ... ]}` — one row per affected
subject (a NOTAM can close several runways → several rows). Same load-bearing instruction as the
other products: "null if not stated; do NOT guess", so abstention vs hallucination stays measurable.
The category itself is fixed per eval record (not emitted by the model); scoring reads it from the
reference. `parse_prediction_notam` canonicalises to the category's fields so the shape matches.
"""

from __future__ import annotations

from avtext.harness.prompt import _extract_json  # reuse the balanced-brace JSON extractor
from avtext.schema.notam import NOTAM_FIELDS

PROMPT_ID_NOTAM = "extract-notam-json-v1"

# One-line framing per category (what the rows represent) — helps the model pick the right subject.
_HINT = {
    "runway": "each affected runway",
    "taxiway": "each affected taxiway",
    "area": "each affected area / airspace restriction",
    "airway": "each affected route segment",
    "stand": "each affected stand",
    "procedure": "each affected instrument procedure",
    "airport": "the airport-level restriction (one row)",
    "light": "each affected lighting system",
    "navigation": "each affected navaid",
}

_TEMPLATE = """\
You are an expert aviation NOTAM decoder. Extract the {category} information from the NOTAM below.

Output ONLY a JSON object of this shape (no prose, no markdown fences):
{{"rows": [ {{...}}, ... ]}}

Emit one row for {hint}. Each row has EXACTLY these keys:
    {fields}

Rules:
- Use exactly those keys; put null for any field the NOTAM does not state. Do NOT guess.
- Values are short strings copied/normalised from the NOTAM (ids, statuses, flight types), or null.
- A NOTAM affecting several subjects -> several rows; one subject -> one row.

NOTAM: {raw}

JSON:"""


def format_prompt_notam(raw_text: str, category: str) -> str:
    fields = ", ".join(NOTAM_FIELDS[category])
    return _TEMPLATE.format(
        category=category, hint=_HINT.get(category, "each affected subject"),
        fields=fields, raw=raw_text,
    )  # fmt: skip


def _canon(v: object) -> str | None:
    """NOTAM values are short strings; stringify + strip, empty -> None (like schema._norm_val)."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def parse_prediction_notam(text: str, category: str) -> dict | None:
    """Model text -> {'rows': [ {field: value|None}, ... ]} in the category's canonical shape, or
    None if unparseable. Keeps only the category's schema fields (extras dropped, missing None)."""
    obj = _extract_json(text)
    if obj is None or not isinstance(obj, dict):
        return None
    fields = NOTAM_FIELDS[category]
    rows = []
    for row in obj.get("rows") or []:
        if isinstance(row, dict):
            rows.append({f: _canon(row.get(f)) for f in fields})
    return {"rows": rows}
