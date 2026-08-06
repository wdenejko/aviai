"""NOTAM runner — drive a served model over eval/notam/* and score it (Phase 7).

Handles BOTH NOTAM tasks, auto-detected from the eval manifest's `kind`:
  • extraction     — category-aware prompt → parse rows → score_notam (row-aligned 5-way), reported
                     per category (value acc / halluc / EM) + row-count accuracy.
  • classification — classify prompt → parse class → classification_metrics (accuracy + macro-F1).
Backend-agnostic (raw /completion via models_notam); deterministic given a deterministic model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from avtext.harness.prompt_notam import (
    PROMPT_ID_NOTAM,
    PROMPT_ID_NOTAM_CLS,
    format_prompt_notam,
    format_prompt_notam_cls,
    parse_prediction_notam,
    parse_prediction_notam_cls,
)
from avtext.harness.score import aggregate, aggregate_by
from avtext.harness.score_notam import classification_metrics, row_count_match, score_notam_record


def _load(eval_path: Path):
    lines = eval_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(x) for x in lines if x.strip()]
    manifest = json.loads((eval_path.parent / "manifest.json").read_text())
    eval_sha = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    return records, manifest, eval_sha


def _map(complete, prompts, concurrency):
    def _safe(p):
        try:
            return complete(p)
        except Exception:
            return ""

    if concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            return list(ex.map(_safe, prompts))
    return [_safe(p) for p in prompts]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def run_extraction(records, complete, concurrency):
    prompts = [format_prompt_notam(r["raw"], r["category"]) for r in records]
    texts = _map(complete, prompts, concurrency)
    scores, preds, raws, invalid_ids, n_rowmatch = [], {}, {}, set(), 0
    for r, t in zip(records, texts, strict=True):
        pred = parse_prediction_notam(t, r["category"])
        if pred is None:  # model output was not valid JSON we could read (mis-escape / truncation)
            invalid_ids.add(r["id"])
        preds[r["id"]] = pred or {}
        raws[r["id"]] = t[:2000]  # keep the head of the raw output so invalids are diagnosable
        n_rowmatch += row_count_match(r["reference"], pred)
        scores.append(
            score_notam_record(r["reference"], pred, id=r["id"], split=r["category"], label="test")
        )
    extra = {"n_invalid": len(invalid_ids), "n_row_match": n_rowmatch}
    return scores, preds, raws, invalid_ids, extra


def run_classification(records, complete, concurrency):
    texts = _map(complete, [format_prompt_notam_cls(r["raw"]) for r in records], concurrency)
    pairs, preds = [], {}
    for r, t in zip(records, texts, strict=True):
        p = parse_prediction_notam_cls(t)
        preds[r["id"]] = p
        pairs.append((r["reference"], p))
    return classification_metrics(pairs), preds


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a model against a frozen NOTAM eval.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--eval", required=True, help="path to eval.jsonl (extraction or classify)")
    ap.add_argument("--out-dir", default="reports/gemma-4/runs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0, help="skip first N records (chunking)")
    ap.add_argument("--out-name", default=None, help="fixed out-dir name (chunks)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args()

    from avtext.harness.models_notam import notam_completer

    eval_path = Path(args.eval)
    records, manifest, eval_sha = _load(eval_path)
    start = args.offset or 0
    records = records[start : start + args.limit if args.limit else None]  # offset then limit
    kind = manifest.get("kind", "extraction")
    prompt_id = PROMPT_ID_NOTAM if kind == "extraction" else PROMPT_ID_NOTAM_CLS
    max_tokens = args.max_tokens or (1024 if kind == "extraction" else 24)
    complete = notam_completer(args.base_url, temperature=args.temperature, max_tokens=max_tokens)
    model_id = args.model_id or Path(args.model).name
    if args.limit and not args.out_name:  # bare --limit = smoke test; a chunk (--out-name) is not
        model_id = f"{model_id} (smoke {args.limit})"

    created = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.out_name:  # deterministic dir so a chunk orchestrator can find + merge the pieces
        out = Path(args.out_dir) / args.out_name
    else:
        out = Path(args.out_dir) / f"{created.replace(':', '').replace('-', '')}-{_slug(model_id)}"
    out.mkdir(parents=True, exist_ok=True)
    header = [
        f"# NOTAM {kind} run — {model_id}",
        "",
        f"- **Date:** {created}  ·  **Prompt:** `{prompt_id}`",
        f"- **Eval:** {manifest['version']} ({kind}) · `{eval_sha[:16]}…` · {len(records)} recs",
        "",
    ]
    run_json = {
        "model_id": model_id, "prompt_id": prompt_id, "kind": kind,
        "eval_version": manifest["version"], "eval_sha256": eval_sha,
        "created_utc": created, "n_records": len(records), "offset": start,
    }  # fmt: skip

    if kind == "extraction":
        scores, preds, raws, invalid_ids, extra = run_extraction(
            records, complete, args.concurrency
        )
        overall = aggregate(scores)
        by_cat = aggregate_by(scores, "split")
        lines = header + [
            f"- **JSON-valid:** {1 - extra['n_invalid'] / len(records):.1%}"
            f"  ·  **row-count acc:** {extra['n_row_match'] / len(records):.1%}",
            "",
            "| category | n | value acc | halluc | EM |",
            "|---|--:|--:|--:|--:|",
            f"| ALL | {overall.n_records} | {overall.recall or 0:.1%}"
            f" | {overall.hallucination_rate or 0:.1%} | {overall.exact_match:.1%} |",
        ]
        for cat, m in by_cat.items():
            lines.append(
                f"| {cat} | {m.n_records} | {m.recall or 0:.1%}"
                f" | {m.hallucination_rate or 0:.1%} | {m.exact_match:.1%} |"
            )
        run_json.update(extra)
        run_json["overall"] = overall.as_dict()
        run_json["by_category"] = {k: v.as_dict() for k, v in by_cat.items()}
        (out / "scores.jsonl").write_text(
            "".join(
                json.dumps({"id": s.id, "category": s.split, "exact_match": s.exact_match,
                            "invalid": s.id in invalid_ids,
                            "outcomes": {k: v.value for k, v in s.outcomes.items()},
                            "prediction": preds.get(s.id, {}),
                            "raw_output": raws.get(s.id, "")}) + "\n"
                for s in scores
            ),
            encoding="utf-8",
        )  # fmt: skip
    else:
        metrics, preds = run_classification(records, complete, args.concurrency)
        lines = header + [
            f"- **accuracy:** {metrics['accuracy']:.1%}  ·  **macro-F1:** {metrics['macro_f1']:.1%}"
            f"  ·  **invalid:** {metrics['invalid']}",
            "",
            "| class | support | precision | recall | f1 |",
            "|---|--:|--:|--:|--:|",
        ]
        for cls, m in sorted(metrics["per_class"].items()):
            lines.append(
                f"| {cls} | {m['support']} | {m['precision']:.1%}"
                f" | {m['recall']:.1%} | {m['f1']:.1%} |"
            )
        run_json["metrics"] = metrics

    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "run.json").write_text(json.dumps(run_json, indent=2) + "\n", encoding="utf-8")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
