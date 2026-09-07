"""Build the LICENSED-ONLY SFT set — the publishable mix (METAR + TAF + NOTAM classification).

The Phase-7 all-products adapter was trained on OpenNOTAM/Knots NOTAM-extraction gold, which has
NO LICENSE (verified 2026-09-07, see DATA_LICENSES.md) — so that adapter is study-only. This mix
keeps only Apache-2.0-compatible sources: IEM/AWC METAR+TAF gold (US public domain) and DEEL-AI
NOTAM classification (MIT). It is the mix for any adapter published on Hugging Face. NOTAM
*extraction* is excluded until upstream grants a license. Same interleave recipe as build_sft_all.

CLI:  uv run python -m avtext.finetune.build_sft_licensed
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

_METAR = Path("data/processed/sft/train_v2.jsonl")
_TAF = Path("data/processed/sft/train_taf.jsonl")
_NOTAM = Path("data/processed/sft/train_notam.jsonl")  # extraction + classification, filtered below
_OUT = Path("data/processed/sft/train_licensed.jsonl")


def _load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _kind(ex: dict) -> str:
    u = ex["messages"][0]["content"]
    if "Decode the TAF" in u:
        return "taf"
    if "Decode the METAR" in u:
        return "metar"
    if "NOTAM classifier" in u:
        return "notam_cls"
    return "notam_ext"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the licensed-only (publishable) SFT set.")
    ap.add_argument("--metar", type=Path, default=_METAR)
    ap.add_argument("--taf", type=Path, default=_TAF)
    ap.add_argument("--notam", type=Path, default=_NOTAM)
    ap.add_argument("--out", type=Path, default=_OUT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # keep ONLY the DEEL-AI classification examples from the NOTAM file (extraction = OpenNOTAM)
    notam_cls = [ex for ex in _load(args.notam) if _kind(ex) == "notam_cls"]
    combined = _load(args.metar) + _load(args.taf) + notam_cls
    random.Random(args.seed).shuffle(combined)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for ex in combined:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    mix = Counter(_kind(ex) for ex in combined)
    assert "notam_ext" not in mix, "unlicensed OpenNOTAM extraction leaked into the licensed mix"
    print(f"wrote {len(combined)} licensed-only SFT examples -> {args.out}")
    print(f"mix: {dict(sorted(mix.items()))}")


if __name__ == "__main__":
    main()
