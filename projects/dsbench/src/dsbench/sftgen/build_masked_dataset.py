"""Build the tokenised training dataset with assistant-only loss labels.

Writes a HF dataset with `input_ids`, `labels` and `num_tokens`, which the patched recipe collator
consumes directly (see patches/recipe-train-assistant-only-loss.patch), plus a JSON build report.
Every block is `--block` tokens. The training kernels are keyed on exact token counts, and 2048,
4096 and 8192 are the keyed lengths (reports/gate-evals/20260928-seq4096-enablement.md).

The default is ADR-004 Revision 2 (`tokenize_masked.pack_thinking`): thinking-on rows, tokenized
as llama.cpp tokenizes them, bin-packed whole into 8192-token blocks. `--legacy` rebuilds the
Gate 1/2 way (stream packing, the GGUF converter's own tokenization), so those runs stay
reproducible.

`num_tokens` is the block length even where a block ends in padding. The collator turns it into the
attention mask, and a mask of all ones keeps the tuned attention path. Padding at the end of a block
cannot reach the tokens before it, because every layer is causal, and its labels are -100.

Runs on the box (dashi): the tokenizer comes from the GGUF base.

    python -m dsbench.sftgen.build_masked_dataset \
        --records ~/benchlab/runs/<run>/mixture.jsonl \
        --out ~/src/transformers5-qwen3.5-recipe/data_tokenized_qwen3.5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer

try:  # installed package (repo checkout)
    from dsbench.sftgen.tokenize_masked import (
        gguf_atomic_tokens,
        match_llama_tokenization,
        pack,
        pack_thinking,
        read_jsonl,
    )
except ImportError:  # standalone on the box, where dsbench is not installed
    from tokenize_masked import (
        gguf_atomic_tokens,
        match_llama_tokenization,
        pack,
        pack_thinking,
        read_jsonl,
    )

MODEL_DIR = "/home/wdenejko/models/qwen3.6"
GGUF = "Qwen3.6-35B-A3B-APEX-I-Mini.gguf"
KEYED_BLOCKS = (2048, 4096, 8192)
# Separates packed rows and pads a block's end: the pretraining document boundary.
END_OF_TEXT = "<|endoftext|>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenise with assistant-only loss labels")
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--block", type=int, default=8192, choices=KEYED_BLOCKS)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--gguf-file", default=GGUF)
    parser.add_argument("--legacy", action="store_true",
                        help="Gate 1/2: stream packing, the GGUF converter's tokenization")
    parser.add_argument("--report", default=None, help="build report (default: <out>.report.json)")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.model_dir, gguf_file=args.gguf_file, local_files_only=True
    )
    records = read_jsonl(args.records)

    if args.legacy:
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        id_blocks, label_blocks, stats = pack(tok, records, args.block, tok.eos_token_id)
        print(f"records            : {stats['records']}")
        print(f"non-prefix-additive: {stats['inexact']}  <- these fall back to full-block loss")
        print(f"blocks ({args.block} tok)  : {stats['blocks']}")
        print(f"dropped all-masked : {stats['dropped_empty_blocks']}")
        print(f"TRAINABLE TOKENS   : {stats['trainable_token_pct']}%")
    else:
        match_llama_tokenization(tok, gguf_atomic_tokens(os.path.join(args.model_dir,
                                                                      args.gguf_file)))
        end = tok.convert_tokens_to_ids(END_OF_TEXT)
        id_blocks, label_blocks, stats = pack_thinking(tok, records, args.block, end, end)
        print(f"records          : {stats['records']}  (rows {stats['rows']})")
        print(f"rejected         : {stats['rejected']}")
        print(f"too long         : {len(stats['too_long'])}")
        print(f"blocks ({args.block})   : {stats['blocks']}  "
              f"(max {stats['rows_per_block_max']} rows per block)")
        print(f"fill             : {stats['fill_pct']}%")
        print(f"trainable tokens : {stats['trainable_token_pct']}%")

    rows = [
        {"input_ids": i, "labels": lab, "num_tokens": len(i)}
        for i, lab in zip(id_blocks, label_blocks, strict=True)
    ]
    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "labels": Sequence(Value("int32")),
            "num_tokens": Value("int32"),
        }
    )
    shutil.rmtree(args.out, ignore_errors=True)
    Dataset.from_list(rows, features=features).save_to_disk(args.out)
    report = args.report or args.out.rstrip("/") + ".report.json"
    with open(report, "w") as handle:
        json.dump({"records_file": args.records, "legacy": args.legacy, **stats}, handle, indent=1)
    print("saved ->", args.out, "| report ->", report)


if __name__ == "__main__":
    main()
