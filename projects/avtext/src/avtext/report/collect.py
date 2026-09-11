"""Collect every run.json under reports/ into one normalized benchmark table.

Each runner (METAR/TAF/NOTAM) writes a slightly different run.json, but they share enough that
one collector can normalize them all into a flat list of comparable rows — the single source the
dashboard reads. Nothing here is thrown away: the raw model_id is kept, and product/variant are
*inferred* (heuristic, from model_id + eval_version) so a new run shows up automatically without
editing this file. Two metric shapes are handled:
  • decode (METAR/TAF/NOTAM-extraction) — run.json has `overall` (a score.Metrics dict)
  • classification (NOTAM)               — run.json has `metrics` (accuracy / macro_f1)

CLI:  uv run python -m avtext.report.collect   ->   reports/gemma-4/benchmarks.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# eval_version -> (product key, human label, task kind). Anything unmapped falls back to the
# raw eval_version so an unexpected eval still lands in the table rather than vanishing.
PRODUCTS: dict[str, tuple[str, str, str]] = {
    "v1": ("metar_v1", "METAR (v1, legacy)", "decode"),
    "v2": ("metar", "METAR", "decode"),
    "taf-v1": ("taf", "TAF", "decode"),
    "notam-v1": ("notam_ext", "NOTAM · extraction", "decode"),
    "notam-cls-v1": ("notam_cls", "NOTAM · classification", "classification"),
}


def _variant(model_id: str) -> str:
    """base (served base) · single (one product) · combined (METAR+TAF) · all (three products,
    Phase-7 29k set) · grown (three products, the S23/S24 49k set — ids carry "all-lg")."""
    m = model_id.lower()
    if "all-lg" in m:
        return "grown"
    if "all-r16" in m or "-all-" in m or "combined-all" in m:
        return "all"
    if "combined" in m:
        return "combined"
    if "base" in m:
        return "base"
    if any(t in m for t in ("r16", "r64", "-lora", " lora")):
        return "single"
    return "base"  # a plain served checkpoint with no adapter marker is the base model


def _model_size(model_id: str) -> str:
    return "E2B" if "e2b" in model_id.lower() else "E4B"


def _rank(model_id: str) -> int | None:
    m = model_id.lower()
    if "r64" in m:
        return 64
    if "r16" in m or "-lora" in m or " lora" in m:
        return 16
    return None


def _normalize(run: dict) -> dict:
    """Pull a single comparable metric block out of whichever shape this run wrote."""
    if "metrics" in run:  # classification
        m = run["metrics"]
        return {
            "kind": "classification",
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "invalid": m.get("invalid"),
            "n_records": m.get("n", run.get("n_records")),
        }
    o = run.get("overall", {})  # decode
    return {
        "kind": "decode",
        "value_acc": o.get("recall"),  # recall == value accuracy (hits / valued)
        "exact_match": o.get("exact_match"),
        "hallucination_rate": o.get("hallucination_rate"),
        "precision": o.get("precision"),
        "f1": o.get("f1"),
        "n_records": o.get("n_records", run.get("n_records")),
        "counts": {
            k: o.get(k) for k in ("hits", "wrong", "abstain", "hallucinate", "true_abstain")
        },
    }


def collect(runs_dir: Path) -> list[dict]:
    rows = []
    for d in sorted(runs_dir.iterdir()):
        rj = d / "run.json"
        if not rj.is_file():
            continue
        run = json.loads(rj.read_text())
        ev = run.get("eval_version", "?")
        product, label, _kind = PRODUCTS.get(ev, (ev, ev, "decode"))
        rows.append(
            {
                "run_dir": d.name,
                "created_utc": run.get("created_utc"),
                "model_id": run.get("model_id", d.name),
                "prompt_id": run.get("prompt_id"),
                "eval_version": ev,
                "eval_sha256": (run.get("eval_sha256") or "")[:16],
                "product": product,
                "product_label": label,
                "variant": _variant(run.get("model_id", "")),
                "model_size": _model_size(run.get("model_id", "")),
                "rank": _rank(run.get("model_id", "")),
                "metrics": _normalize(run),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect all run.json into one benchmark table.")
    ap.add_argument("--runs-dir", type=Path, default=Path("reports/gemma-4/runs"))
    ap.add_argument("--out", type=Path, default=Path("reports/gemma-4/benchmarks.json"))
    args = ap.parse_args()

    rows = collect(args.runs_dir)
    args.out.write_text(json.dumps({"runs": rows}, indent=2) + "\n", encoding="utf-8")
    by_product: dict[str, int] = {}
    for r in rows:
        by_product[r["product"]] = by_product.get(r["product"], 0) + 1
    print(f"collected {len(rows)} runs -> {args.out}")
    for p, n in sorted(by_product.items()):
        print(f"  {p:12} {n}")


if __name__ == "__main__":
    main()
