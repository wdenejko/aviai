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


def run_eval(
    predict: Predictor,
    *,
    model_id: str,
    eval_path: Path = _EVAL,
    prompt_id: str = PROMPT_ID,
    decode_params: dict | None = None,
) -> RunResult:
    """Run `predict` over every eval record and score it. Deterministic given a deterministic
    predictor — the eval order is fixed and we never sample here."""
    lines = eval_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    eval_sha = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    manifest = json.loads((eval_path.parent / "manifest.json").read_text())

    scores: list[RecordScore] = []
    n_invalid = 0
    for rec in records:
        try:
            pred = predict(rec["raw"])
        except Exception:  # a backend crash on one report must not sink the whole run
            pred = None
        if pred is None:
            n_invalid += 1
            pred = {}  # scored as abstention everywhere
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
    lines = [
        f"# Run report — {r.model_id}",
        "",
        f"- **Date:** {r.created_utc}",
        f"- **Model:** `{r.model_id}`  ·  **Prompt:** `{r.prompt_id}`",
        f"- **Eval:** {r.eval_version} · `sha256:{r.eval_sha256[:16]}…`",
        f"- **Decode:** `{json.dumps(r.decode_params) if r.decode_params else 'n/a'}`",
        f"- **Records:** {n}  ·  **JSON-valid:** {1 - r.n_invalid / n:.1%}"
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
                    }
                )
                + "\n"
            )
    return out
