# ruff: noqa: E501 -- the strings below reproduce the recipe's source line for line (text to match
# and text to insert), so their line lengths are the recipe's, not this repo's.
"""Write ~/benchlab/scripts/audit_long_seq.py: the recipe's training-step audit, extended past 2048.

The recipe's audit refuses sequences over 2048 because its dataset rows are 2048-token blocks.
The copy glues consecutive full rows (chunks of one tokenized stream) into longer sequences and
keeps the gradient-completeness check for every length >= 2048 (a longer sequence starts with the
same row, so it routes to a superset of the experts the 2048 audit reaches).
"""
from pathlib import Path

src = Path.home() / "src/transformers5-qwen3.5-recipe/audit_qwen3_5_training_step.py"
t = src.read_text()
old_check = """    if sequence_length <= 1 or sequence_length > 2048:
        raise ValueError(
            "The fixed Qwen dataset supports sequence lengths in [2,2048]."
        )"""
new_check = """    # dashi long-seq audit: rows are 2048-token blocks, so a longer sequence glues PER
    # consecutive full rows together (rows are chunks of one tokenized stream).
    if sequence_length <= 1 or (sequence_length > 2048 and sequence_length % 2048):
        raise ValueError("sequence length must be <= 2048 or a multiple of 2048")
    per = max(1, sequence_length // 2048)"""
old_select = """    if row_start < 0 or row_start + batch_size > len(dataset):
        raise IndexError(
            f"requested rows [{row_start},{row_start + batch_size}) outside "
            f"dataset of {len(dataset)} rows"
        )
    selected = dataset.select(range(row_start, row_start + batch_size))
    rows = [selected[index] for index in range(batch_size)]
"""
new_select = """    if row_start < 0 or row_start + batch_size * per > len(dataset):
        raise IndexError(
            f"requested rows [{row_start},{row_start + batch_size * per}) outside "
            f"dataset of {len(dataset)} rows"
        )
    selected = dataset.select(range(row_start, row_start + batch_size * per))
    parts = [selected[index] for index in range(batch_size * per)]
    if per > 1 and any(int(row["num_tokens"]) != 2048 for row in parts):
        raise ValueError("gluing rows needs full 2048-token rows (no padding mid-sequence)")
    rows = [
        {
            "input_ids": sum((parts[b * per + j]["input_ids"] for j in range(per)), []),
            "num_tokens": sum(int(parts[b * per + j]["num_tokens"]) for j in range(per)),
        }
        for b in range(batch_size)
    ]
"""
old_req = "    require_complete = args.sequence_length == 2048\n"
new_req = "    require_complete = args.sequence_length >= 2048  # longer = a superset of the 2048 routes\n"
for old, new in ((old_check, new_check), (old_select, new_select), (old_req, new_req)):
    assert t.count(old) == 1, old[:60]
    t = t.replace(old, new)
header = (
    "# dashi copy of the recipe audit (ADR-001 retrain, 2026-09-28): the same step, but a sequence\n"
    "# longer than 2048 glues consecutive 2048-token dataset rows. Run it with the recipe on\n"
    "# PYTHONPATH and --dataset-dir set (the default is relative to this file). Source: the recipe\n"
    "# audit_qwen3_5_training_step.py with its box patch (QWEN35_ATTN_IMPL); made by\n"
    "# make_audit_long_seq.py.\n"
)
dst = Path.home() / "benchlab/scripts/audit_long_seq.py"
dst.write_text(header + t)
print("wrote", dst)
