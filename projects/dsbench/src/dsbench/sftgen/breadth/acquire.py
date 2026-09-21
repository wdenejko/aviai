"""Acquire breadth/replay slices from the registered public sources (ADR-001 §B.4).

Pipeline per source (streamed, so we never download a multi-GB corpus for a small pilot slice):

    stream rows -> row_ok pre-filter -> normalize to the training shape -> drop over-long traces
    (ADR filter: > 8k tokens) -> decontaminate vs dsbench -> stop at the per-source token cap.

Each kept record is written immediately (crash-safe) with provenance `meta`; a manifest records the
per-source yield + why rows were dropped. `datasets` is imported lazily inside `acquire_source`, so
`--list` and imports work without the dep; only a real fetch needs `uv run --with datasets`.

    uv run --with datasets python -m dsbench.sftgen.breadth.acquire --list
    uv run --with datasets python -m dsbench.sftgen.breadth.acquire \
        --source jupyter_agent --cap-tokens 300000 --out-dir data/sft
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from dsbench.sftgen.breadth.sources import SOURCES, SOURCES_BY_KEY, Source, public_sources
from dsbench.sftgen.decontaminate import _row_text, build_denylist, scan_text

MAX_TRACE_TOKENS = 8000   # ADR-001: "drop traces > 8k tok"
_DEFAULT_CAP = 300_000    # per-source token cap (pilot surplus; the assembler samples down)
_DEFAULT_MAX_SCAN = 40_000  # rows to stream before giving up on reaching the cap


def approx_tokens(rec: dict) -> int:
    """chars/3.5 over the trainable text (message content + tool-call arguments). Cheap + consistent
    with the rest of sftgen; the authoritative count happens at assembly with the real tokenizer."""
    chars = 0
    for m in rec.get("messages", []):
        chars += len(m.get("content") or "")
        for tc in (m.get("tool_calls") or []):
            chars += len((tc.get("function") or {}).get("arguments") or "")
    return round(chars / 3.5)


def acquire_source(src: Source, *, cap_tokens: int, max_scan: int, deny, sink) -> dict:
    from datasets import load_dataset  # lazy: only a real fetch needs the heavy dep

    stats: dict[str, Any] = {
        "source": src.key, "hf_id": src.hf_id, "bucket": src.bucket, "licence": src.licence,
        "teacher": src.teacher, "redistributable": src.redistributable,
        "scanned": 0, "emitted": 0, "tokens": 0,
        "skipped_row_ok": 0, "skipped_normalize": 0, "skipped_long": 0, "skipped_contam": 0,
        "contam_by_rule": {},
    }
    ds = load_dataset(src.hf_id, name=src.config, split=src.split, streaming=True)
    for row in ds:
        if stats["scanned"] >= max_scan:
            break
        stats["scanned"] += 1
        if not src.row_ok(row):
            stats["skipped_row_ok"] += 1
            continue
        rec = src.normalize(row)
        if rec is None:
            stats["skipped_normalize"] += 1
            continue
        toks = approx_tokens(rec)
        if toks == 0 or toks > MAX_TRACE_TOKENS:
            stats["skipped_long"] += 1
            continue
        reason = scan_text(_row_text(rec), deny)
        if reason is not None:
            stats["skipped_contam"] += 1
            rule = reason["rule"]
            stats["contam_by_rule"][rule] = stats["contam_by_rule"].get(rule, 0) + 1
            continue
        rec["meta"].update({
            "source_key": src.key, "bucket": src.bucket, "hf_id": src.hf_id,
            "licence": src.licence, "teacher": src.teacher,
            "redistributable": src.redistributable, "approx_tokens": toks,
        })
        sink(rec)
        stats["emitted"] += 1
        stats["tokens"] += toks
        if stats["tokens"] >= cap_tokens:
            break
    return stats


def _print_registry() -> None:
    print(f"{'key':16s} {'bucket':13s} {'gated':5s} {'redist':6s} {'teacher':16s} hf_id")
    for s in SOURCES:
        print(f"{s.key:16s} {s.bucket:13s} {str(s.gated):5s} {str(s.redistributable):6s} "
              f"{s.teacher:16s} {s.hf_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Acquire breadth/replay slices (ADR-001 B.4)")
    ap.add_argument("--list", action="store_true", help="print the source registry and exit")
    ap.add_argument("--source", default="", help="one source key")
    ap.add_argument("--bucket", default="", help="all public sources in this bucket")
    ap.add_argument("--all-public", action="store_true", help="every non-gated source")
    ap.add_argument("--cap-tokens", type=int, default=_DEFAULT_CAP)
    ap.add_argument("--max-scan", type=int, default=_DEFAULT_MAX_SCAN)
    ap.add_argument("--out-dir", default="", help="write breadth_<key>.jsonl + manifest here")
    args = ap.parse_args()

    if args.list:
        _print_registry()
        return

    if args.source:
        picked = [SOURCES_BY_KEY[args.source]]
    elif args.bucket:
        picked = [s for s in public_sources() if s.bucket == args.bucket]
    elif args.all_public:
        picked = public_sources()
    else:
        raise SystemExit("pick --source / --bucket / --all-public (or --list)")

    deny = build_denylist()
    manifest: dict[str, Any] = {"max_trace_tokens": MAX_TRACE_TOKENS, "cap_tokens": args.cap_tokens,
                                "sources": []}
    for src in picked:
        path = f"{args.out_dir}/breadth_{src.key}.jsonl" if args.out_dir else None
        fh = open(path, "w") if path else None

        def _sink(rec, _fh=fh) -> None:
            if _fh:
                _fh.write(json.dumps(rec) + "\n")
                _fh.flush()

        try:
            stats = acquire_source(src, cap_tokens=args.cap_tokens, max_scan=args.max_scan,
                                   deny=deny, sink=_sink)
        finally:
            if fh:
                fh.close()
        manifest["sources"].append(stats)
        print(f"[{src.key}] emitted {stats['emitted']} rows / ~{stats['tokens']:,} tok "
              f"(scanned {stats['scanned']}; long {stats['skipped_long']}, "
              f"contam {stats['skipped_contam']}, row_ok {stats['skipped_row_ok']}, "
              f"norm {stats['skipped_normalize']})")

    if args.out_dir:
        with open(f"{args.out_dir}/breadth_manifest.json", "w") as mf:
            json.dump(manifest, mf, indent=2)


if __name__ == "__main__":
    main()
