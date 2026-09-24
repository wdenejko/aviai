"""Unit tests for Gate-2 eval-bucket membership (pure; no data files, no model)."""
from __future__ import annotations

from dsbench.sftgen.eval_buckets_gate2 import content_key, pilot_recall


def _row(user: str, answer: str, **meta) -> dict:
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": answer}], "meta": meta}


def test_content_key_ignores_meta_and_dict_key_order():
    """Pools and mixtures annotate meta differently; identity must come from content alone."""
    a = _row("q", "a", mix_pool="tulu3")
    b = {"meta": {"seq_id": 7}, "messages": [{"content": "q", "role": "user"},
                                             {"content": "a", "role": "assistant"}]}
    assert content_key(a) == content_key(b)


def test_content_key_separates_rows_sharing_a_prompt():
    """Target C reuses one prompt per family; rows differ only after the first user turn."""
    assert content_key(_row("same prompt", "run 1")) != content_key(_row("same prompt", "run 2"))


def test_pilot_recall_reports_a_leak():
    """A trained row whose serialisation drifted is NOT found -- and must show up as a miss."""
    pool = [_row("q1", "a1"), _row("q2", "a2")]
    pilot = [_row("q1", "a1"), _row("q2", "a2 (renormalised)")]
    assert pilot_recall(pool, pilot) == (1, 2)


def test_pilot_recall_counts_distinct_rows():
    """Duplicate pool rows must not inflate recall past the pilot's own row count."""
    pool = [_row("q", "a"), _row("q", "a")]
    assert pilot_recall(pool, [_row("q", "a")]) == (1, 1)
