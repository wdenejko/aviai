"""CPU check of packed rows' data path: build_masked_dataset's `seq_lens` through the patched
recipe collator, block by block.

The collator is the patched train_qwen3_5_35b.py's own function, compiled from its source: an
import would load the recipe's modules, and with them AITER and Triton. For every block of a
dataset built by build_masked_dataset.py, it checks what would reach the model:
- the segments are the build report's (`block_seq_lens`), in order, and cover the block;
- position ids restart at 0 at every segment and count up within it;
- every boundary follows a separator, the attention mask is all ones, and the labels are the
  dataset's.
The rows it counts must be the report's `rows_packed`. Then two batches the collator must
handle: two blocks at once (refused: FLA's varlen rule flattens the batch) and a block without
`seq_lens`, Gate 1/2's format (passed through as before).

On the box, in the ftgguf toolbox (CPU only), with packed_rows.py beside it:

    python check_collator.py DATASET_DIR PATCHED_TRAIN_SCRIPT

The build report is read from DATASET_DIR.report.json, build_masked_dataset's default.
"""

from __future__ import annotations

import ast
import itertools
import json
import sys

import torch
from datasets import load_from_disk
from packed_rows import add_segments
from transformers import AutoTokenizer, default_data_collator

MODEL_DIR = "/home/wdenejko/models/qwen3.6"
GGUF = "Qwen3.6-35B-A3B-APEX-I-Mini.gguf"


def collator(path: str):
    with open(path) as handle:
        tree = ast.parse(handle.read())
    fn = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == "fixed_length_lm_collator")
    namespace = {"torch": torch, "default_data_collator": default_data_collator,
                 "add_segments": add_segments}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), namespace)
    return namespace["fixed_length_lm_collator"]


def main() -> None:
    dataset_dir, train_script = sys.argv[1], sys.argv[2]
    with open(dataset_dir.rstrip("/") + ".report.json") as handle:
        report = json.load(handle)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, gguf_file=GGUF, local_files_only=True)
    separator = tok.convert_tokens_to_ids("<|endoftext|>")
    dataset = load_from_disk(dataset_dir)
    collate = collator(train_script)
    assert len(dataset) == len(report["block_seq_lens"])
    rows = 0
    for example, segments in zip(dataset, report["block_seq_lens"], strict=True):
        assert list(example["seq_lens"]) == segments
        batch = collate([dict(example)])
        cu = batch["cu_seq_lens_q"].tolist()
        assert cu == [0, *itertools.accumulate(segments)], (cu, segments)
        assert cu[-1] == batch["input_ids"].shape[1] == batch["position_ids"].shape[1]
        assert torch.equal(batch["cu_seq_lens_k"], batch["cu_seq_lens_q"])
        assert batch["max_length_q"] == batch["max_length_k"] == max(segments)
        assert torch.equal(batch["position_ids"][0], torch.cat([torch.arange(n) for n in segments]))
        ids = batch["input_ids"][0].tolist()
        assert all(ids[s - 1] == separator for s in cu[1:])  # each segment ends on one
        assert bool(batch["attention_mask"].all())
        assert batch["labels"][0].tolist() == list(example["labels"])
        padded = ids[cu[-2]] == separator  # a padding segment is separators from its start
        rows += len(segments) - padded
    assert rows == report["rows_packed"], (rows, report["rows_packed"])
    try:
        collate([dict(dataset[0]), dict(dataset[1])])
        raise AssertionError("two blocks in one batch were accepted")
    except ValueError:
        pass
    legacy = {k: v for k, v in dataset[0].items() if k != "seq_lens"}
    assert "cu_seq_lens_q" not in collate([legacy])
    print(f"ok: {len(dataset)} blocks, {rows} rows (the report's), separator {separator}; "
          "two blocks in a batch refused; a block without seq_lens unchanged")


if __name__ == "__main__":
    main()
