"""Incremental prediction checkpoints for long eval runs (S24).

Why this exists: a harness run predicts thousands of records and, until now, only wrote anything at
the very end. On 2026-09-09 dashi hard-reset 15 h into the 5,294-record TAF eval and every
prediction died with the process. With a checkpoint file each prediction is appended the moment it
lands, and a rerun of the same model on the same eval loads the file and predicts only what is
missing — a reset costs minutes, not the stage.

Design choices worth knowing:
- The file is keyed by (model_id, eval sha) in a header line. A different model or a changed eval
  never resumes from stale predictions — the store is silently started fresh.
- Failed predictions (None — a backend exception on one record) are NOT persisted, so a transient
  server failure is retried on resume instead of being frozen into the result as an abstention.
- `predict_all` returns predictions in eval order whatever the completion order under a threadpool,
  so scores and the saved predictions stay deterministic regardless of concurrency or resumption.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


class PartialStore:
    def __init__(self, path: Path, *, model_id: str, eval_sha: str) -> None:
        self.path = Path(path)
        self.model_id = model_id
        self.eval_sha = eval_sha
        self._preds: dict[str, dict] = {}
        self._fh = None

    def load(self) -> dict[str, dict]:
        """Predictions already on disk for this (model, eval); {} if none or stale."""
        self._preds = {}
        if not self.path.exists():
            return {}
        header, *rows = self.path.read_text(encoding="utf-8").splitlines()
        try:
            h = json.loads(header) if header.strip() else {}
        except json.JSONDecodeError:
            h = {}
        if h.get("model_id") != self.model_id or h.get("eval_sha") != self.eval_sha:
            return {}
        for line in rows:
            if line.strip():
                o = json.loads(line)
                self._preds[o["id"]] = o["pred"]
        return dict(self._preds)

    def append(self, id: str, pred: dict) -> None:
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fresh = not self._preds  # nothing loaded -> (re)start the file with our header
            self._fh = self.path.open("w" if fresh else "a", encoding="utf-8")
            if fresh:
                self._fh.write(
                    json.dumps({"model_id": self.model_id, "eval_sha": self.eval_sha}) + "\n"
                )
        self._fh.write(json.dumps({"id": id, "pred": pred}, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._preds[id] = pred

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def predict_all(
    items: list[tuple[str, Any]],
    predict: Callable[[Any], dict | None],
    *,
    store: PartialStore | None = None,
    concurrency: int = 1,
) -> list[dict | None]:
    """Run `predict` over (id, input) pairs, skipping ids the store already holds, checkpointing
    each successful prediction, and returning results in the order of `items`."""
    done: dict[str, dict | None] = dict(store.load()) if store is not None else {}
    todo = [(i, x) for i, x in items if i not in done]

    def _one(item: tuple[str, Any]) -> tuple[str, dict | None]:
        i, x = item
        return i, predict(x)

    def _record(i: str, p: dict | None) -> None:
        done[i] = p
        if store is not None and p is not None:
            store.append(i, p)

    try:
        if concurrency > 1 and todo:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                for fut in as_completed([ex.submit(_one, it) for it in todo]):
                    _record(*fut.result())
        else:
            for it in todo:
                _record(*_one(it))
    finally:
        if store is not None:
            store.close()
    return [done.get(i) for i, _ in items]
