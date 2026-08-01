"""TAF runner — drive a predictor over eval/taf, score it, write a reproducible report (Phase 6).

The TAF sibling of harness/runner.py: same Predictor seam ((raw) -> nested decode | None), same
frozen-eval discipline and reproducibility block, but scores with score_taf (period-aligned) and
adds a TAF-specific structural metric — period-count accuracy (did the model get the number of
change groups right?). Backend-agnostic; point --base-url at any served model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from avtext.harness.prompt_taf import PROMPT_ID_TAF
from avtext.harness.score import Metrics, RecordScore, aggregate
from avtext.harness.score_taf import period_count_match, score_taf_record
from avtext.harness.stats import bootstrap_ci, fmt_ci

Predictor = Callable[[str], dict | None]

_EVAL = Path("eval/taf/v1/eval.jsonl")
_RUNS = Path("reports/gemma-4/runs")


@dataclass
class RunResult:
    model_id: str
    prompt_id: str
    eval_version: str
    eval_sha256: str
    decode_params: dict
    created_utc: str
    n_invalid: int
    n_period_match: int
    scores: list[RecordScore] = field(default_factory=list)
    predictions: dict[str, dict] = field(default_factory=dict)


def run_eval_taf(
    predict: Predictor, *, model_id: str, eval_path: Path = _EVAL, prompt_id: str = PROMPT_ID_TAF,
    decode_params: dict | None = None, limit: int | None = None, concurrency: int = 1,
) -> RunResult:  # fmt: skip
    """Run `predict` over every TAF eval record and score it. Deterministic given a deterministic
    predictor (fixed eval order, no sampling here)."""
    lines = eval_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    if limit is not None:
        records = records[:limit]
    eval_sha = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    manifest = json.loads((eval_path.parent / "manifest.json").read_text())

    def _safe(raw: str) -> dict | None:
        try:
            return predict(raw)
        except Exception:  # a backend crash on one report must not sink the run
            return None

    if concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            preds = list(ex.map(_safe, (r["raw"] for r in records)))
    else:
        preds = [_safe(r["raw"]) for r in records]

    scores: list[RecordScore] = []
    predictions: dict[str, dict] = {}
    n_invalid = 0
    n_period_match = 0
    for rec, pred in zip(records, preds, strict=True):
        if pred is None:
            n_invalid += 1
        predictions[rec["id"]] = pred or {}
        n_period_match += period_count_match(rec["reference"], pred)
        scores.append(
            score_taf_record(
                rec["reference"], pred, id=rec["id"], split=rec["split"], label=rec["label"]
            )
        )
    return RunResult(
        model_id=model_id,
        prompt_id=prompt_id,
        eval_version=manifest["version"],
        eval_sha256=eval_sha,
        decode_params=decode_params or {},
        created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_invalid=n_invalid,
        n_period_match=n_period_match,
        scores=scores,
        predictions=predictions,
    )


def _cells(scores: list[RecordScore]) -> dict[tuple[str, str], list[RecordScore]]:
    cells: dict[tuple[str, str], list[RecordScore]] = {}
    for s in scores:
        cells.setdefault((s.split, s.label), []).append(s)
    return cells


def _render_report(r: RunResult) -> str:
    overall = aggregate(r.scores)
    n = overall.n_records
    valid = (1 - r.n_invalid / n) if n else 0.0
    pcm = (r.n_period_match / n) if n else 0.0
    lines = [
        f"# TAF run report — {r.model_id}",
        "",
        f"- **Date:** {r.created_utc}",
        f"- **Model:** `{r.model_id}`  ·  **Prompt:** `{r.prompt_id}`",
        f"- **Eval:** {r.eval_version} · `sha256:{r.eval_sha256[:16]}…`",
        f"- **Decode:** `{json.dumps(r.decode_params) if r.decode_params else 'n/a'}`",
        f"- **Records:** {n}  ·  **JSON-valid:** {valid:.1%} ({r.n_invalid} unusable)"
        f"  ·  **period-count acc:** {pcm:.1%}",
        "",
        "Columns: **value acc** = recall (hits / values that existed), 95% bootstrap CI;"
        " **halluc** = fabricated / truly-absent; **EM** = whole-forecast exact match.",
        "",
        "## Overall",
        "| scope | n | value acc [95% CI] | halluc | EM |",
        "|---|--:|--|--:|--:|",
        _row("all", r.scores),
        "",
        "## By split × label",
        "| split | label | n | value acc [95% CI] | halluc | EM |",
        "|---|---|--:|--|--:|--:|",
    ]
    cells = _cells(r.scores)
    for split, lab in sorted(cells):
        m = aggregate(cells[(split, lab)])
        rec_ci = fmt_ci(*bootstrap_ci(cells[(split, lab)], lambda x: aggregate(x).recall))
        lines.append(
            f"| {split} | {lab} | {m.n_records} | {rec_ci} |"
            f" {fmt_ci(m.hallucination_rate, None, None)} | {m.exact_match:.0%} |"
        )
    lines += [
        "",
        "## Reproducibility",
        f"- eval `{r.eval_version}` — sha256 `{r.eval_sha256}`",
        f"- prompt `{r.prompt_id}`  ·  model `{r.model_id}`  ·  decode `{r.decode_params}`",
        "",
    ]
    return "\n".join(lines)


def _row(name: str, scores: list[RecordScore]) -> str:
    m = aggregate(scores)
    rec_ci = fmt_ci(*bootstrap_ci(scores, lambda x: aggregate(x).recall))
    return f"| {name} | {m.n_records} | {rec_ci} | {fmt_ci(m.hallucination_rate, None, None)} | {m.exact_match:.0%} |"  # noqa: E501


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def write_report(result: RunResult, base: Path = _RUNS) -> Path:
    stamp = result.created_utc.replace(":", "").replace("-", "")
    out = base / f"{stamp}-{_slug(result.model_id)}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(_render_report(result), encoding="utf-8")
    overall: Metrics = aggregate(result.scores)
    (out / "run.json").write_text(
        json.dumps(
            {
                "model_id": result.model_id,
                "prompt_id": result.prompt_id,
                "eval_version": result.eval_version,
                "eval_sha256": result.eval_sha256,
                "decode_params": result.decode_params,
                "created_utc": result.created_utc,
                "n_invalid": result.n_invalid,
                "n_period_match": result.n_period_match,
                "overall": overall.as_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    with (out / "scores.jsonl").open("w", encoding="utf-8") as fh:
        for s in result.scores:
            fh.write(
                json.dumps(
                    {
                        "id": s.id,
                        "split": s.split,
                        "label": s.label,
                        "exact_match": s.exact_match,
                        "outcomes": {k: v.value for k, v in s.outcomes.items()},
                        "prediction": result.predictions.get(s.id, {}),
                    }
                )
                + "\n"
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a model against frozen eval/taf.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=str(_RUNS))
    ap.add_argument("--completion", action="store_true", help="use raw /completion, not chat")
    ap.add_argument("--eval", default=str(_EVAL))
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()

    if args.completion:
        from avtext.harness.models_taf import completion_predictor_taf

        predict = completion_predictor_taf(
            args.base_url, temperature=args.temperature, max_tokens=args.max_tokens
        )
    else:
        from avtext.harness.models_taf import http_predictor_taf

        predict = http_predictor_taf(
            args.base_url, args.model, temperature=args.temperature, max_tokens=args.max_tokens
        )
    model_id = args.model_id or Path(args.model).name
    if args.limit:
        model_id = f"{model_id} (smoke {args.limit})"
    result = run_eval_taf(
        predict, model_id=model_id, eval_path=Path(args.eval),
        decode_params={"temperature": args.temperature, "max_tokens": args.max_tokens},
        limit=args.limit, concurrency=args.concurrency,
    )  # fmt: skip
    out = write_report(result, base=Path(args.out_dir))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
