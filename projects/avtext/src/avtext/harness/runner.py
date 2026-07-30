"""Runner — drive a predictor over eval/v1, score it, write a reproducible report (Phase 3).

A `Predictor` is just `(raw) -> canonical field dict | None`. That one-line contract is the
whole seam between "how a decode is produced" (a parser, a local LLM, a frontier API) and
"how it's judged" (everything in score.py/stats.py). None means the model emitted no usable
JSON — recorded as an invalid output, and scored as abstention on every field.

Definition of done for Phase 3 is here: `write_report` emits `reports/runs/<id>/` with the
metrics AND a reproducibility block — eval sha256, prompt id, model id, decode params — so
any number in the report can be re-derived from a clean clone. The harness is frozen before
training; this is what a training run will be measured against.
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

from avtext.consensus.vote import flatten
from avtext.harness.prompt import PROMPT_ID
from avtext.harness.score import Metrics, RecordScore, aggregate, score_record
from avtext.harness.stats import bootstrap_ci, fmt_ci
from avtext.oracles.base import Oracle

Predictor = Callable[[str], dict | None]

_EVAL = Path("eval/v1/eval.jsonl")
_RUNS = Path("reports/runs")


def oracle_predictor(oracle: Oracle) -> Predictor:
    """Wrap a parser as a predictor — the baseline/wiring backend. A parser abstains (None)
    when it can't parse, and never fabricates; that's the reference point an LLM must beat."""

    def predict(raw: str) -> dict | None:
        r = oracle.decode(raw)
        return flatten(r.obs) if r.ok and r.obs is not None else None

    return predict


@dataclass
class RunResult:
    model_id: str
    prompt_id: str
    eval_version: str
    eval_sha256: str
    decode_params: dict
    created_utc: str
    n_invalid: int  # predictions that were unusable (no JSON) — a formatting failure
    scores: list[RecordScore] = field(default_factory=list)
    predictions: dict[str, dict] = field(default_factory=dict)  # id -> canonical pred


def run_eval(
    predict: Predictor,
    *,
    model_id: str,
    eval_path: Path = _EVAL,
    prompt_id: str = PROMPT_ID,
    decode_params: dict | None = None,
    limit: int | None = None,
    concurrency: int = 1,
) -> RunResult:
    """Run `predict` over every eval record and score it. Deterministic given a deterministic
    predictor — the eval order is fixed and we never sample here. `limit` truncates to the
    first N records for a smoke test (a limited run is marked in the model_id by the caller)."""
    lines = eval_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    if limit is not None:
        records = records[:limit]
    eval_sha = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    manifest = json.loads((eval_path.parent / "manifest.json").read_text())

    def _safe(raw: str) -> dict | None:
        try:
            return predict(raw)
        except Exception:  # a backend crash on one report must not sink the whole run
            return None

    # Predict concurrently — the v2 eval is 6,200 records (10x v1); a threadpool over a
    # --parallel llama.cpp server cuts wall-clock ~Nx. Order is preserved (ex.map), so
    # scoring and the saved predictions stay deterministic regardless of concurrency.
    if concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            preds = list(ex.map(_safe, (r["raw"] for r in records)))
    else:
        preds = [_safe(r["raw"]) for r in records]

    scores: list[RecordScore] = []
    predictions: dict[str, dict] = {}
    n_invalid = 0
    for rec, pred in zip(records, preds, strict=True):
        if pred is None:
            n_invalid += 1
            pred = {}  # scored as abstention everywhere
        predictions[rec["id"]] = pred
        scores.append(
            score_record(
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
        scores=scores,
        predictions=predictions,
    )


def _cells(scores: list[RecordScore]) -> dict[tuple[str, str], list[RecordScore]]:
    cells: dict[tuple[str, str], list[RecordScore]] = {}
    for s in scores:
        cells.setdefault((s.split, s.label), []).append(s)
    return cells


def _row(name: str, scores: list[RecordScore]) -> str:
    m = aggregate(scores)
    rec_ci = fmt_ci(*bootstrap_ci(scores, lambda x: aggregate(x).recall))
    hr = fmt_ci(m.hallucination_rate, None, None)
    return f"| {name} | {m.n_records} | {rec_ci} | {hr} | {m.exact_match:.0%} |"


def _render_report(r: RunResult) -> str:
    overall = aggregate(r.scores)
    n = overall.n_records
    valid_frac = (1 - r.n_invalid / n) if n else 0.0  # guard: empty eval must not divide by zero
    lines = [
        f"# Run report — {r.model_id}",
        "",
        f"- **Date:** {r.created_utc}",
        f"- **Model:** `{r.model_id}`  ·  **Prompt:** `{r.prompt_id}`",
        f"- **Eval:** {r.eval_version} · `sha256:{r.eval_sha256[:16]}…`",
        f"- **Decode:** `{json.dumps(r.decode_params) if r.decode_params else 'n/a'}`",
        f"- **Records:** {n}  ·  **JSON-valid:** {valid_frac:.1%}"
        f" ({r.n_invalid} unusable)",
        "",
        "Columns: **value acc** = recall (hits / values that existed), with 95% bootstrap CI;"
        " **halluc** = fabricated / truly-absent (the safety metric); **EM** = whole-record.",
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
    for split, label in sorted(cells):
        m = aggregate(cells[(split, label)])
        rec_ci = fmt_ci(*bootstrap_ci(cells[(split, label)], lambda x: aggregate(x).recall))
        hr = fmt_ci(m.hallucination_rate, None, None)
        lines.append(
            f"| {split} | {label} | {m.n_records} | {rec_ci} | {hr} | {m.exact_match:.0%} |"
        )
    lines += [
        "",
        "## Reproducibility",
        "Every number above re-derives from:",
        f"- eval set `{r.eval_version}` — sha256 `{r.eval_sha256}`",
        f"- prompt `{r.prompt_id}`  ·  model `{r.model_id}`  ·  decode `{r.decode_params}`",
        "",
        "Re-run: `make harness MODEL=<id>` (rebuilds this report from the frozen eval).",
        "",
    ]
    return "\n".join(lines)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def write_report(result: RunResult, base: Path = _RUNS) -> Path:
    """Write reports/runs/<id>/ — report.md (human), run.json (repro block + overall metrics),
    scores.jsonl (per-record outcomes, for drill-down)."""
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
    """`python -m avtext.harness.runner` — score a model server against frozen eval/v1.

    Backend-agnostic: point --base-url at any OpenAI-compatible server (mlx_vlm.server for
    the local Gemma, ollama, llama-server, LM Studio). The model is never trained here; this
    is measurement against a frozen set."""
    from avtext.harness.models import http_predictor

    ap = argparse.ArgumentParser(description="Score a model against frozen eval/v1.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", required=True, help="served model id / path")
    ap.add_argument("--model-id", default=None, help="label for the report (default: basename)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on the first N records")
    # Runs are namespaced by model family (reports/gemma-4/runs, reports/gemma-3/runs) so a
    # model-family pivot never silently mixes results — default keeps the pre-split layout.
    ap.add_argument("--out-dir", default=str(_RUNS), help="where to write reports/<...>/runs")
    # Finetuned models overfit to the exact training prompt; the chat-template render can drift
    # (minja vs HF). --completion sends the raw training wrapper to /completion so serve==train.
    ap.add_argument("--completion", action="store_true", help="use raw /completion, not chat")
    ap.add_argument("--eval", default=str(_EVAL), help="path to eval.jsonl (default eval/v1)")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel requests (v2: 8+)")
    args = ap.parse_args()

    if args.completion:
        from avtext.harness.models import completion_predictor

        predict = completion_predictor(
            args.base_url, temperature=args.temperature, max_tokens=args.max_tokens
        )
    else:
        predict = http_predictor(
            args.base_url, args.model, temperature=args.temperature, max_tokens=args.max_tokens
        )
    model_id = args.model_id or Path(args.model).name
    if args.limit:
        model_id = f"{model_id} (smoke {args.limit})"
    result = run_eval(
        predict,
        model_id=model_id,
        eval_path=Path(args.eval),
        decode_params={"temperature": args.temperature, "max_tokens": args.max_tokens},
        limit=args.limit,
        concurrency=args.concurrency,
    )
    out = write_report(result, base=Path(args.out_dir))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
