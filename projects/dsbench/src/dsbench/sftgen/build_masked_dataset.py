"""Build the tokenised training dataset with assistant-only loss labels.

Writes a HF dataset with `input_ids`, `labels` and `num_tokens`, which the patched recipe collator
consumes directly (see patches/recipe-train-assistant-only-loss.patch). Block length MUST stay at
the MMQ bundle's compiled geometry (2048) or training fails on an unsupported deployment key.

Runs on the box (dashi): the tokenizer comes from the GGUF base.

    python -m dsbench.sftgen.build_masked_dataset \
        --records ~/pilot_mixture.jsonl \
        --out ~/src/transformers5-qwen3.5-recipe/data_tokenized_qwen3.5
"""

from __future__ import annotations

import argparse
import shutil

from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer

try:  # installed package (repo checkout)
    from dsbench.sftgen.tokenize_masked import pack, read_jsonl
except ImportError:  # standalone on the box, where dsbench is not installed
    from tokenize_masked import pack, read_jsonl

MODEL_DIR = "/home/wdenejko/models/qwen3.6"
GGUF = "Qwen3.6-35B-A3B-APEX-I-Mini.gguf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenise with assistant-only loss labels")
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--block", type=int, default=2048)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--gguf-file", default=GGUF)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.model_dir, gguf_file=args.gguf_file, local_files_only=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    records = read_jsonl(args.records)
    id_blocks, label_blocks, stats = pack(tok, records, args.block, tok.eos_token_id)

    print(f"records            : {stats['records']}")
    print(f"non-prefix-additive: {stats['inexact']}  <- these fall back to full-block loss")
    print(f"blocks ({args.block} tok)  : {stats['blocks']}")
    print(f"dropped all-masked : {stats['dropped_empty_blocks']}")
    print(f"TRAINABLE TOKENS   : {stats['trainable_token_pct']}%  (was 100% before this change)")

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
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
