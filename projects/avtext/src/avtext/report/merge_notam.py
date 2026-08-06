"""Merge chunked NOTAM extraction runs into one run.json (Session 22 / Phase 7 fix).

Long NOTAM extraction evals must be chunked with a server restart between chunks (the ROCm
llama-server degrades into reserved-token spam after ~1000 heavy-generation records — see
eval_notam_chunked.sh). Each chunk writes a normal scores.jsonl + run.json for its slice; this
stitches them back into a single run.json/report.md identical in shape to an unchunked extraction
run, so the collector/dashboard treat it like any other run.

CLI:  uv run python -m avtext.report.merge_notam --chunks <dir-of-chunk-dirs> --model-id <id>
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from avtext.harness.score import Outcome, RecordScore, aggregate, aggregate_by


def _load_chunk(chunk_dir: Path) -> tuple[list[RecordScore], list[dict], dict]:
    run = json.loads((chunk_dir / "run.json").read_text())
    scores, rawrows = [], []
    for line in (chunk_dir / "scores.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        scores.append(
            RecordScore(
                id=d["id"], split=d["category"], label="test",
                outcomes={f: Outcome(v) for f, v in d["outcomes"].items()},
                exact_match=d["exact_match"],
            )  # fmt: skip
        )
        rawrows.append(d)
    return scores, rawrows, run


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge chunked NOTAM extraction runs.")
    ap.add_argument("--chunks", type=Path, required=True, help="dir containing per-chunk subdirs")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/gemma-4/runs"))
    ap.add_argument("--model-id", required=True)
    args = ap.parse_args()

    chunk_dirs = sorted(d for d in args.chunks.iterdir() if (d / "scores.jsonl").is_file())
    if not chunk_dirs:
        raise SystemExit(f"no chunk dirs with scores.jsonl under {args.chunks}")

    all_scores, all_rows, n_invalid, n_row_match, meta = [], [], 0, 0, {}
    seen: set[str] = set()
    for cd in chunk_dirs:
        scores, rawrows, run = _load_chunk(cd)
        meta = meta or run  # keep prompt_id / eval_version / eval_sha256 from the first chunk
        n_row_match += run.get("n_row_match", 0)
        for s, d in zip(scores, rawrows, strict=True):
            if s.id in seen:  # chunks are disjoint slices; guard anyway
                continue
            seen.add(s.id)
            all_scores.append(s)
            all_rows.append(d)
            n_invalid += 1 if d.get("invalid") else 0

    overall = aggregate(all_scores)
    by_cat = aggregate_by(all_scores, "split")
    created = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.out_dir / f"{created.replace(':', '').replace('-', '')}-{args.model_id}-merged"
    out.mkdir(parents=True, exist_ok=True)

    run_json = {
        "model_id": args.model_id, "prompt_id": meta.get("prompt_id"), "kind": "extraction",
        "eval_version": meta.get("eval_version"), "eval_sha256": meta.get("eval_sha256"),
        "created_utc": created, "n_records": len(all_scores), "n_chunks": len(chunk_dirs),
        "n_invalid": n_invalid, "n_row_match": n_row_match,
        "overall": overall.as_dict(),
        "by_category": {k: v.as_dict() for k, v in by_cat.items()},
    }  # fmt: skip
    (out / "run.json").write_text(json.dumps(run_json, indent=2) + "\n", encoding="utf-8")
    (out / "scores.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in all_rows), encoding="utf-8"
    )
    lines = [
        f"# NOTAM extraction (merged {len(chunk_dirs)} chunks) — {args.model_id}",
        "",
        f"- **Eval:** {meta.get('eval_version')} · {len(all_scores)} recs · "
        f"JSON-valid {1 - n_invalid / max(len(all_scores), 1):.1%} · "
        f"row-count acc {n_row_match / max(len(all_scores), 1):.1%}",
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
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"merged {len(chunk_dirs)} chunks ({len(all_scores)} recs) -> {out}")
    print(f"  value={overall.recall or 0:.1%} halluc={overall.hallucination_rate or 0:.1%} "
          f"EM={overall.exact_match:.1%} invalid={n_invalid}")  # fmt: skip


if __name__ == "__main__":
    main()
