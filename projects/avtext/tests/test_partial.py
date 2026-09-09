"""PartialStore / predict_all — the S24 crash-safety contract for long eval runs."""

from pathlib import Path

from avtext.harness.partial import PartialStore, predict_all


def test_store_round_trip_and_stale_header(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    st = PartialStore(p, model_id="m", eval_sha="abc")
    assert st.load() == {}
    st.append("r1", {"a": 1})
    st.append("r2", {"b": 2})
    st.close()
    assert PartialStore(p, model_id="m", eval_sha="abc").load() == {"r1": {"a": 1}, "r2": {"b": 2}}
    # a different eval (or model) must never resume from these predictions
    assert PartialStore(p, model_id="m", eval_sha="zzz").load() == {}
    assert PartialStore(p, model_id="other", eval_sha="abc").load() == {}
    # ...and starts the file over on first append
    st2 = PartialStore(p, model_id="m", eval_sha="zzz")
    st2.load()
    st2.append("r9", {"z": 9})
    st2.close()
    assert PartialStore(p, model_id="m", eval_sha="zzz").load() == {"r9": {"z": 9}}


def test_predict_all_resumes_only_missing_and_retries_failures(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    items = [("a", "A"), ("b", "B"), ("c", "C")]
    calls: list[str] = []

    def flaky(x: str):
        calls.append(x)
        return None if x == "C" else {"v": x.lower()}

    out = predict_all(
        items, flaky, store=PartialStore(p, model_id="m", eval_sha="s"), concurrency=2
    )
    assert out == [{"v": "a"}, {"v": "b"}, None]  # eval order, failure -> None
    assert sorted(calls) == ["A", "B", "C"]

    calls.clear()

    def ok(x: str):
        calls.append(x)
        return {"v": x}

    out2 = predict_all(items, ok, store=PartialStore(p, model_id="m", eval_sha="s"), concurrency=1)
    assert calls == ["C"]  # a, b resumed from disk; the failed one is retried, not frozen
    assert out2 == [{"v": "a"}, {"v": "b"}, {"v": "C"}]


def test_predict_all_without_store_preserves_order_under_concurrency() -> None:
    items = [(str(i), i) for i in range(20)]
    out = predict_all(items, lambda x: {"sq": x * x}, store=None, concurrency=4)
    assert out == [{"sq": i * i} for i in range(20)]
