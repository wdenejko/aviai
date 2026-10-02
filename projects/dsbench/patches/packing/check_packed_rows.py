"""Check, on the GPU, that a block with its rows' boundaries trains each row as if it were alone.

ADR-004 Revision 2, action item 1. One 8,192-token block of real thinking-on rows goes through the
recipe's training model, set up as train_qwen3_5_35b.py sets it up (the GGUF base, rank-4 LoRA with
B = 0, the aiter attention kernel, FLA), three ways:
- `packed`: with its rows' boundaries (packed_rows.py). Position ids restart at each row, and
  cu_seq_lens reach attention, the GatedDeltaNet rule and FLA's convolution;
- `today`: as Gate 1 and Gate 2 trained, without them;
- `alone`: each row by itself at the start of a block, with its boundaries; padding fills the rest.

Each goes through a forward pass, for the last hidden state at every row token, and one training
step, for the loss and the LoRA-B gradients (LoRA-A's are zero while B is). The rows alone are the
reference:
- with boundaries, each packed row's hidden states should match its row alone about as closely
  as the packed block matches itself run again (`noise`);
- the block's loss should be the rows' losses weighted by their labelled tokens, and its
  gradients the same weighted sum of theirs;
- without boundaries, the rows after the first see the rows before them. The first row sees
  nothing else either way, so its `today` figure compares only the kernels: the
  variable-length ones against those Gate 1 and 2 trained with. On 2026-10-02 they differed
  only by the convolution's rounding, which the network carried to 29% (check_kernels.py,
  reports/gate-evals/20261002-packed-rows-check.md).
`head` covers each row's first 16 tokens, where the convolution's leak would show.

On the box, in the ftgguf toolbox, during a GPU window (patches/packing/window.sh):

    python check_packed_rows.py OUT.json TARGET_C.jsonl PILOT.jsonl [--dry-run]

TARGET_C.jsonl gives the shortest kept rows of two families (tool calls, several turns), and
PILOT.jsonl up to two more (one turn) while the block has room. `--dry-run` stops before the
model loads, once the rows are tokenized and packed and their arguments checked: CPU only.
Needs tokenize_masked.py and packed_rows.py beside it, and the recipe and AITER on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from packed_rows import configure_qwen35_packed_conv, segment_kwargs
from tokenize_masked import (
    IGNORE,
    RowRejected,
    gguf_atomic_tokens,
    match_llama_tokenization,
    thinking_record,
)
from transformers import AutoTokenizer

MODEL_DIR = Path.home() / "models/qwen3.6"
GGUF = "Qwen3.6-35B-A3B-APEX-I-Mini.gguf"
BLOCK = 8192
MIN_PADDING = 32  # most blocks end in padding, so this one does too
HEAD = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "gate_proj", "up_proj",
                  "in_proj_qkv", "in_proj_z", "out_proj", "experts"]  # train_qwen3_5_35b.py's


def read(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encoded(tok, record: dict) -> dict | None:
    try:
        ids, labels = thinking_record(tok, record)
    except RowRejected:
        return None
    return {"id": record["meta"]["id"], "ids": ids, "labels": labels}


def family(row_id: str) -> str:
    return row_id.rsplit("-", 1)[0]  # C-mlc_churn_rare-221 -> C-mlc_churn_rare


def selection(record: dict) -> dict:
    return record["meta"].get("selection", {})


def pick_rows(tok, target_c: list[dict], pilot: list[dict]) -> list[dict]:
    """The shortest kept Target C rows of two families, then pilot rows while the block has room."""
    rows: list[dict] = []
    kept = [r for r in target_c if selection(r).get("kept", True)]
    for record in sorted(kept, key=lambda r: selection(r).get("served_tokens", 0)):
        if family(record["meta"]["id"]) not in {family(row["id"]) for row in rows}:
            row = encoded(tok, record)
            if row:
                rows.append(row)
        if len(rows) == 2:
            break
    room = BLOCK - MIN_PADDING - sum(len(row["ids"]) + 1 for row in rows)
    for record in pilot:
        if len(rows) == 4:
            break
        row = encoded(tok, record)
        if row and len(row["ids"]) + 1 <= room:
            rows.append(row)
            room -= len(row["ids"]) + 1
    return rows


def block_of(rows: list[dict], end: int) -> tuple[list[int], list[int], list[int]]:
    """`pack_thinking`'s layout: each row and its separator, then padding to BLOCK."""
    ids: list[int] = []
    labels: list[int] = []
    for row in rows:
        ids += row["ids"] + [end]
        labels += row["labels"] + [IGNORE]
    padding = BLOCK - len(ids)
    seq_lens = [len(row["ids"]) + 1 for row in rows] + ([padding] if padding else [])
    return ids + [end] * padding, labels + [IGNORE] * padding, seq_lens


def inputs(ids: list[int], labels: list[int], seq_lens: list[int] | None, device: str) -> dict:
    """What the patched collator gives the model; `seq_lens=None` is today's collator."""
    batch = {"input_ids": torch.tensor([ids], dtype=torch.long),
             "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
             "labels": torch.tensor([labels], dtype=torch.long)}
    if seq_lens is not None:
        batch.update(segment_kwargs(seq_lens))
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    """`a` against the reference `b`."""
    a, b = a.float().flatten(), b.float().flatten()
    diff = a - b
    return {"rel_l2": float(diff.norm() / b.norm()), "max_abs": float(diff.abs().max()),
            "cos": float(torch.nn.functional.cosine_similarity(a, b, dim=0))}


def group(name: str) -> str:
    for key, label in (("experts", "experts"), ("linear_attn", "gdn"), ("self_attn", "attention")):
        if key in name:
            return label
    return "shared_mlp"


def grads_vs(grads: dict, reference: dict) -> dict:
    out = {}
    for label in sorted({group(name) for name in reference}):
        names = [n for n in reference if group(n) == label and n in grads]
        out[label] = compare(torch.cat([grads[n].float().flatten() for n in names]),
                             torch.cat([reference[n].float().flatten() for n in names]))
        out[label]["tensors"] = len(names)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("target_c")
    ap.add_argument("pilot")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    report: dict = {"block": BLOCK, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def save() -> None:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, gguf_file=GGUF, local_files_only=True)
    match_llama_tokenization(tok, gguf_atomic_tokens(str(MODEL_DIR / GGUF)))
    end = tok.convert_tokens_to_ids("<|endoftext|>")
    rows = pick_rows(tok, read(args.target_c), read(args.pilot))
    if len(rows) < 2:
        raise SystemExit(f"{len(rows)} row(s) found: nothing to pack")
    ids, labels, seq_lens = block_of(rows, end)
    report["rows"] = [{"id": row["id"], "tokens": len(row["ids"]),
                       "labelled": sum(x != IGNORE for x in row["labels"])} for row in rows]
    # The arguments themselves: the segments cover the block, positions restart at every row,
    # and every row after the first follows a separator.
    kwargs = segment_kwargs(seq_lens)
    cu = kwargs["cu_seq_lens_q"].tolist()
    starts = cu[:-1]
    assert len(ids) == len(labels) == cu[-1] == BLOCK, (len(ids), cu[-1])
    assert all(int(kwargs["position_ids"][0, s]) == 0 for s in starts)
    assert all(ids[s - 1] == end for s in starts[1:])
    report["seq_lens"] = seq_lens
    report["cu_seq_lens"] = cu
    save()
    print(json.dumps(report, indent=1), flush=True)
    if args.dry_run:
        return

    import fla
    import transformers
    from attention_aiter_tuning import configure_qwen35_flash_attention_2
    from fast_lora import register_fast_lora
    from fast_moe_lora import register_fast_moe_lora
    from fast_moe_ranking import configure_fast_moe_ranking
    from fla_tuning import configure_qwen35_fla
    from gguf_dequant_compile import configure_compiled_gguf_dequantize
    from gguf_liger_loss import apply_gguf_liger_fused_linear_cross_entropy
    from peft import LoraConfig, TaskType, get_peft_model
    from qwen3_5_fused_norms import configure_qwen35_fused_norms
    from transformers import AutoModelForCausalLM

    report["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                          "fla": fla.__version__}
    torch.manual_seed(19260817)
    configure_compiled_gguf_dequantize()
    configure_qwen35_flash_attention_2()
    configure_qwen35_fla()
    configure_qwen35_packed_conv()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, gguf_file=GGUF, gguf_mmap_policy="release", local_files_only=True,
        dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation=os.environ.get("QWEN35_ATTN_IMPL", "flash_attention_2"))
    report["attn_implementation"] = str(model.config._attn_implementation)
    # Boundaries reach attention only through flash attention's variable-length path.
    if "flash" not in report["attn_implementation"]:
        save()
        raise SystemExit(f"not a flash attention kernel: {report['attn_implementation']}")
    model.config.use_cache = False
    model.config.output_router_logits = False
    model.config.router_aux_loss_coef = 0.0
    configure_fast_moe_ranking(model)
    configure_qwen35_fused_norms(model)
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=TARGET_MODULES, r=4,
                      lora_alpha=4, use_rslora=False)
    register_fast_lora(lora, model)
    register_fast_moe_lora(lora, model, expert_prior="qwen-learned")
    model = get_peft_model(model, lora, autocast_adapter_dtype=False)
    apply_gguf_liger_fused_linear_cross_entropy(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    report["load_s"] = round(time.time() - t0, 1)
    save()
    text_model = model.get_base_model().model
    lora_b = [(n, p) for n, p in model.named_parameters() if p.requires_grad and ".lora_B" in n]

    def hidden(batch: dict) -> torch.Tensor:
        with torch.no_grad():
            out = text_model(**{k: v for k, v in batch.items() if k != "labels"}, use_cache=False)
        return out.last_hidden_state[0].float().cpu()

    def step(batch: dict) -> tuple[float, dict]:
        model.zero_grad(set_to_none=True)
        loss = model(**batch, use_cache=False).loss
        loss.backward()
        return float(loss.detach()), {n: p.grad.detach().cpu() for n, p in lora_b
                                      if p.grad is not None}

    def timed(name: str, fn, batch: dict):
        t = time.time()
        result = fn(batch)
        report.setdefault("seconds", {})[name] = round(time.time() - t, 1)
        save()
        print(f"{name}: {report['seconds'][name]}s", flush=True)
        return result

    device = "cuda:0"
    packed = inputs(ids, labels, seq_lens, device)
    today = inputs(ids, labels, None, device)
    alone = [inputs(*block_of([row], end), device) for row in rows]

    # The first packed pass also autotunes the variable-length kernels, so `again` is the timing.
    h_packed = timed("hidden_packed", hidden, packed)
    h_today = timed("hidden_today", hidden, today)
    h_alone = [timed(f"hidden_alone_{i}", hidden, batch) for i, batch in enumerate(alone)]
    h_again = timed("hidden_packed_again", hidden, packed)
    report["hidden"] = {"noise": compare(h_again, h_packed), "rows": []}
    for i, row in enumerate(rows):
        n, s = len(row["ids"]), starts[i]
        entry = {"id": row["id"]}
        for name, h in (("packed", h_packed), ("today", h_today)):
            entry[name] = compare(h[s : s + n], h_alone[i][:n])
            entry[name + "_head"] = compare(h[s : s + HEAD], h_alone[i][:HEAD])
        report["hidden"]["rows"].append(entry)
    save()
    del h_packed, h_today, h_alone, h_again

    loss_packed, g_packed = timed("step_packed", step, packed)
    loss_today, g_today = timed("step_today", step, today)
    total = sum(row["labelled"] for row in report["rows"])
    losses: list[float] = []
    g_rows: dict = {}
    for i, batch in enumerate(alone):
        loss, grads = timed(f"step_alone_{i}", step, batch)
        losses.append(loss)
        weight = report["rows"][i]["labelled"] / total
        for name, grad in grads.items():
            g_rows[name] = g_rows.get(name, 0) + weight * grad.float()
        del grads
    loss_again, g_again = timed("step_packed_again", step, packed)
    report["loss"] = {
        "packed": loss_packed, "packed_again": loss_again, "today": loss_today, "alone": losses,
        "alone_weighted": sum(row["labelled"] / total * x
                              for row, x in zip(report["rows"], losses, strict=True))}
    report["grads"] = {"noise": grads_vs(g_again, g_packed),
                       "packed": grads_vs(g_packed, g_rows), "today": grads_vs(g_today, g_rows)}
    report["memory_gib"] = {"allocated_peak": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                            "reserved_peak": round(torch.cuda.max_memory_reserved() / 2**30, 1)}
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()

    print(f"\nhidden rel_l2 vs the row alone (noise {report['hidden']['noise']['rel_l2']:.2e}):")
    for entry in report["hidden"]["rows"]:
        print(f"  {entry['id']:<48} packed {entry['packed']['rel_l2']:.2e} "
              f"(head {entry['packed_head']['rel_l2']:.2e})  "
              f"today {entry['today']['rel_l2']:.2e} (head {entry['today_head']['rel_l2']:.2e})")
    print(f"loss: packed {loss_packed:.5f}, rows weighted {report['loss']['alone_weighted']:.5f}, "
          f"today {loss_today:.5f}, packed again {loss_again:.5f}")
    for label in report["grads"]["packed"]:
        print(f"grads {label:<10} cos packed {report['grads']['packed'][label]['cos']:.5f}  "
              f"today {report['grads']['today'][label]['cos']:.5f}  "
              f"noise {report['grads']['noise'][label]['cos']:.5f}")


if __name__ == "__main__":
    main()
