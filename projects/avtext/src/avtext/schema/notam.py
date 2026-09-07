"""Canonical NOTAM extraction schema (Phase 7).

NOTAMs invert the METAR/TAF setup: parsers fail the free-text E-field, so there is no parser
consensus. The **gold is given** — the expert annotations in the Knots dataset (github.com/
Estrellajer/Knots) — and the task is *category-specific structured extraction* from the raw E-field.
This module is the contract: which fields each category yields, and the shape they take.

Design decisions (each earned from inspecting the data 2026-08-04):

1. NINE CATEGORIES, TWO SHAPES, unified as ROWS. Six categories are row-based (one NOTAM can affect
   several runways/taxiways → several rows); three are flat (a single extraction). We represent BOTH
   as `rows: list[dict]` — a flat category is simply a one-row list — so the scorer aligns rows
   uniformly (the same idea as TAF change-group periods).

2. ENGLISH ONLY — NO CHINESE DATAPOINTS (project requirement). The source is ADCC/China-annotated,
   so some raw text and one field are Chinese. We enforce cleanliness two ways: (a) `area_type` is a
   6-VALUE Chinese ENUM (not free text), so it is REMAPPED to English (AREA_TYPE_REMAP) and kept —
   recovering the area category with its type; (b) any record still containing a Chinese character
   in `raw_text` or any field value is dropped at ingest (`has_chinese`). The created dataset
   therefore contains no Chinese anywhere.

3. VALUES ARE CATEGORICAL STRINGS. Unlike METAR's numerics, NOTAM fields are short strings / small
   enums ("clsd", "international,domestic,regional", a runway id "13") or null. We keep them as
   strings (light-normalised: stripped, ""→None) rather than over-typing — faithful to the gold.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator


class NotamCategory(StrEnum):
    AIRPORT = "airport"
    AIRWAY = "airway"
    AREA = "area"
    LIGHT = "light"
    NAVIGATION = "navigation"
    PROCEDURE = "procedure"
    RUNWAY = "runway"
    RVR = "rvr"  # runway visual range sensor status
    STANDARD = "standard"  # approach-minima changes
    STAND = "stand"
    TAXIWAY = "taxiway"


# Canonical fields per category, in output order. Chinese-only fields are EXCLUDED (area_type).
# Field names are normalised to snake_case (see _KNOTS_KEY_MAP for the source-key mapping).
NOTAM_FIELDS: dict[str, tuple[str, ...]] = {
    "runway": (
        "airport", "runway", "status_type", "affect_region", "flight_type",
        "ppr", "aip", "tora", "toda", "asda", "lda", "distance_chg",
    ),
    "taxiway": ("airport", "taxiway", "status_type", "section", "intersection_with"),
    "area": ("area_type", "area_summary", "height_detail", "atc", "fpl"),  # area_type remapped
    "rvr": (
        "airport", "runway",
        "touchdown_zone_unavailable", "midpoint_unavailable", "stop_end_unavailable",
    ),
    "standard": (
        "airport", "runway", "procedure_name", "approach_type",
        "aircraft_category", "minima_type", "minima_value", "notam_type",
    ),
    "airway": (
        "route", "start", "end", "directional", "height_detail", "atc", "fpl", "change_info",
    ),
    "stand": ("airport", "stand_split_info", "stand_status"),
    "procedure": ("airport", "runway", "procedure_type", "procedure_name", "chart", "aip"),
    "airport": (
        "airport", "restriction_type", "flight_type", "affect_region", "diversion_restriction",
        "aip", "ppr", "fuel", "industrialaction", "powersupply",
    ),
    "light": (
        "airport", "runway", "lightcategory", "ilscategory",
        "unavailable_downgrade", "als", "distance", "percentage",
    ),
    "navigation": ("airport", "runway", "navaid_id", "navaid_type"),
}  # fmt: skip


# Categories where a NOTAM yields MANY rows; the rest are flat (exactly one row).
def notam_json_schema(category: str) -> dict:
    """JSON schema for one category's extraction output (for grammar-constrained decoding).

    llama.cpp's `/completion` compiles a `json_schema` into GBNF and constrains sampling to it,
    so the model cannot emit malformed JSON — the ~13.5% invalid-JSON tail on the free-text
    categories (`area`, `airway`) goes to zero. Shape mirrors what `parse_prediction_notam`
    expects: {"rows": [{field: string|null, ...}]} with exactly this category's fields.
    """
    fields = NOTAM_FIELDS[category]
    row = {
        "type": "object",
        "properties": {f: {"type": ["string", "null"]} for f in fields},
        "required": list(fields),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"rows": {"type": "array", "items": row}},
        "required": ["rows"],
        "additionalProperties": False,
    }


ROW_CATEGORIES = frozenset({"runway", "taxiway", "area", "airway", "stand", "procedure"})

# Knots source field name -> our canonical name (fixes caps / slashes). area_type is intentionally
# absent from every category's field tuple, so it is dropped even though Knots emits it.
_KNOTS_KEY_MAP = {
    "Chart": "chart",
    "restriction_Type": "restriction_type",
    "Diversion_Restriction": "diversion_restriction",
    "unavailable/downgrade": "unavailable_downgrade",
}

# area_type is a fixed 6-value Chinese enum -> English (the ONLY Chinese in the area category).
# Remapping it recovers all of area with its type field, Chinese-free. Any value NOT here stays
# Chinese and its record is dropped by has_chinese (a safety net for unexpected values).
AREA_TYPE_REMAP = {
    "区域激活": "activation",
    "多边形": "polygon",
    "圆": "circle",
    "圆弧多边形": "arc",
    "线段外扩": "line_buffer",
    "扇形": "sector",
}


def has_chinese(x: object) -> bool:
    """True if any CJK character appears anywhere in x (str / list / dict, recursively).
    The guard that keeps the created dataset Chinese-free (raw_text AND every kept field value)."""
    if x is None:
        return False
    if isinstance(x, dict):
        return any(has_chinese(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return any(has_chinese(v) for v in x)
    return any("一" <= c <= "鿿" for c in str(x))


def _norm_val(v: object) -> str | None:
    """Light value normalisation: stringify, strip, empty -> None. Keeps case (ids matter)."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def normalize_row(category: str, raw_row: dict) -> dict[str, str | None]:
    """Map one source row to a canonical {field: value} dict for `category`, keeping only that
    category's schema fields (stray keys dropped) and remapping the area_type Chinese enum."""
    fields = NOTAM_FIELDS[category]
    remapped = {_KNOTS_KEY_MAP.get(k, k): v for k, v in raw_row.items()}
    row = {f: _norm_val(remapped.get(f)) for f in fields}
    if category == "area" and row.get("area_type") is not None:
        row["area_type"] = AREA_TYPE_REMAP.get(row["area_type"], row["area_type"])
    return row


# ── NOTAM classification (a SECOND, independent task from DEEL-AI/NOTAM, MIT) ─────────────────────
# A different task shape from extraction: assign one raw NOTAM to one of 13 top-level classes. Kept
# here in the NOTAM domain but scored/prompted separately (classification, not field extraction).
NOTAM_CLASSES = (
    "Airspaces", "Communication_and_Radar", "GPS", "Landing_Navaids", "Obstacles", "Parking",
    "Procedure", "Runway", "Services_and_Facilities", "Signs_and_Lights", "Taxiway",
    "Terminal_or_Enroute_Navaids", "Wildlife",
)  # fmt: skip


class NotamClassification(BaseModel):
    """A NOTAM labelled with one top-level class (DEEL-AI). The classification task's record."""

    model_config = ConfigDict(extra="forbid")

    id: str
    raw_text: str
    label: str  # one of NOTAM_CLASSES


class NotamExtraction(BaseModel):
    """A decoded NOTAM — the canonical record the model must produce and the scorer grades against.
    `rows` are category-specific field dicts (a flat category has exactly one row)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: NotamCategory
    raw_text: str
    rows: list[dict[str, str | None]]

    @field_validator("rows")
    @classmethod
    def _rows_match_category_fields(cls, rows, info):
        cat = info.data.get("category")
        if cat is None:
            return rows
        allowed = set(NOTAM_FIELDS[cat.value])
        for row in rows:
            extra = set(row) - allowed
            if extra:
                raise ValueError(f"{cat.value} row has non-schema fields: {sorted(extra)}")
        return rows
