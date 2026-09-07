"""Standard transformers+peft+trl LoRA finetune — reproducible replacement for the lost unsloth
train_lora.py (Session 23).

The dashi `~/ft` unsloth env was lost with no recipe, so this rebuilds the trainer on plain Hugging
Face PEFT — no unsloth, so the recipe reproduces anywhere (the resume-grade path). Same LoRA config
(rank, alpha=2r, the 7 proj modules), same chat-template formatting (`apply_chat_template` over the
{messages:[user,assistant]} SFT rows), and same optimizer/schedule as the original, so new adapters
stay comparable to the Phase 6/7 results. Output is a PEFT adapter that convert_lora_to_gguf turns
into the GGUF our llama.cpp harness serves.

Run inside the reconstructed ROCm venv:  HSA_OVERRIDE_GFX_VERSION=11.0.0 ~/fttorch/bin/python \
    train_lora_peft.py --model unsloth/gemma-4-E4B-it --out ~/ft/gemma-4/<name> --data <sft.jsonl> \
    --rank 16 --epochs 1 --batch 6 --max-seq 2048 [--packing]
"""

import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch._dynamo  # noqa: F401  (module-level: importing inside main() would shadow `torch`)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer


class LengthGroupedSFTTrainer(SFTTrainer):
    """SFTTrainer whose train sampler batches examples of similar token length together.

    trl 1.x dropped TrainingArguments.group_by_length, so we inject the sampler directly. On gfx1151
    this is the single biggest lever measured: random batches mixing METAR (~460 tok) with TAF
    (~1100+ tok) pad every row to the longest, and 49% of linear compute was padding. Length-grouped
    batches bring that to ~1% (≈2x less linear work, ≈3x less attention work)."""

    def _get_train_sampler(self, train_dataset=None):
        ds = train_dataset if train_dataset is not None else self.train_dataset
        if ds is None or "input_ids" not in ds.column_names:
            return super()._get_train_sampler(train_dataset)
        lengths = [len(x) for x in ds["input_ids"]]
        gen = torch.Generator()
        gen.manual_seed(self.args.seed)
        return LengthGroupedSampler(
            self.args.per_device_train_batch_size, lengths=lengths, generator=gen
        )

# E4B is multimodal; its vision/audio towers use Gemma4ClippableLinear (a non-nn.Linear wrapper
# PEFT cannot LoRA) and we only decode TEXT. This regex matches only the 258 text-tower proj layers
# (model.language_model.*), so the adapter is text-only — matching the text GGUF the harness serves.
TEXT_TARGETS = (
    r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plain-PEFT LoRA finetune (no unsloth).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--grad-accum", type=int, default=2)
    # gfx1151 perf: with no working flash-attn, packing flattens the batch into one long sequence
    # and SDPA's math backend does O(flattened_len^2) attention — catastrophically slow (S23:
    # 310s/step). Non-packed + dynamic padding keeps short METAR/NOTAM rows short. And with ~100 GiB
    # of unified RAM free, gradient checkpointing (a compute-for-memory trade) is usually a net loss
    # here — --no-grad-checkpointing removes the recompute. Bigger --batch fills the under-used iGPU.
    ap.add_argument(
        "--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--packing", action="store_true")
    # S24 gfx1151 speed levers (measured on dashi):
    #  --group-by-length: random batches of mixed METAR(~460 tok)/TAF(~1100+) pad every row to the
    #    longest -> 49% of linear compute was padding. Length-grouped batches cut that to ~1%
    #    (~2x linear, ~3x attention work).
    #  --compile: torch.compile via the Trainer (inductor) fused the ~35% unfused elementwise ops
    #    for a measured 1.42x on a real step. Sequence length varies per batch, so we rely on
    #    dynamo's automatic dynamic shapes + a raised recompile cap instead of recompiling per shape.
    ap.add_argument("--group-by-length", action="store_true")
    ap.add_argument("--compile", action="store_true")
    # Resilience: gfx1151's experimental AOTriton attention can page-fault intermittently (S24: a
    # GPU "Memory access fault" mid-run, not reproducible, not thermal). A fault kills the process,
    # so long runs checkpoint every --save-steps and --resume picks up the latest checkpoint;
    # run_resilient.sh loops the two so a fault costs minutes, not the night.
    ap.add_argument("--save-steps", type=int, default=0, help="checkpoint every N steps (0=off)")
    ap.add_argument("--resume", action="store_true", help="resume from latest checkpoint in --out")
    a = ap.parse_args()
    if a.compile:  # variable seq lens -> automatic dynamic shapes; allow a few recompiles
        torch._dynamo.config.cache_size_limit = 64

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.config.use_cache = False
    if a.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # so grads flow with checkpointing on a frozen base
    model = get_peft_model(
        model,
        LoraConfig(
            r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.0, bias="none",
            target_modules=TEXT_TARGETS, task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=a.data, split="train")
    ds = ds.map(
        lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)},
        remove_columns=ds.column_names,
    )

    cfg = SFTConfig(
        dataset_text_field="text", packing=a.packing, max_length=a.max_seq,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
        warmup_steps=10, num_train_epochs=a.epochs, max_steps=a.max_steps,
        learning_rate=a.lr, logging_steps=5, optim="adamw_torch", weight_decay=0.01,
        lr_scheduler_type="linear", seed=42, output_dir=a.out, report_to="none",
        dataset_num_proc=1, bf16=True, gradient_checkpointing=a.grad_checkpointing,
        torch_compile=a.compile,
        save_strategy="steps" if a.save_steps else "no", save_steps=a.save_steps or 500,
        save_total_limit=2,
    )
    trainer_cls = LengthGroupedSFTTrainer if a.group_by_length else SFTTrainer
    trainer = trainer_cls(model=model, train_dataset=ds, args=cfg, processing_class=tok)
    last = get_last_checkpoint(a.out) if (a.resume and os.path.isdir(a.out)) else None
    if last:
        print("RESUMING_FROM", last, flush=True)
    trainer.train(resume_from_checkpoint=last)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("SAVED_ADAPTER", a.out)


if __name__ == "__main__":
    main()
