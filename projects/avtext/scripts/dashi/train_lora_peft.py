"""Standard transformers+peft+trl LoRA finetune — reproducible replacement for the lost unsloth
train_lora.py (Session 23).

The dashi `~/ft` unsloth env was lost with no recipe, so this rebuilds the trainer on plain Hugging
Face PEFT — no unsloth, so the recipe reproduces anywhere (the resume-grade path). Same LoRA config
(rank, alpha=2r, the 7 proj modules), same chat-template formatting (`apply_chat_template` over the
{messages:[user,assistant]} SFT rows), and same optimizer/schedule as the original, so new adapters
stay comparable to the Phase 6/7 results. Output is a PEFT adapter that convert_lora_to_gguf turns
into the GGUF our llama.cpp harness serves.

Run inside the reconstructed ROCm venv/toolbox (never set HSA_OVERRIDE_GFX_VERSION on gfx1151):
    ~/fttorch/bin/python train_lora_peft.py --model unsloth/gemma-4-E4B-it --out ~/ft/gemma-4/<name> \
    --data <sft.jsonl> --rank 16 --epochs 1 --batch 6 --grad-accum 2 --max-seq 2048 \
    --group-by-length --compile --ckpt-above <N>   # see docs/STRIX_HALO_TRAINING_SPEED.md
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


class AvtextSFTTrainer(SFTTrainer):
    """SFTTrainer with the two gfx1151 levers that need trainer internals.

    1. Length-grouped batches (`group_by_length`): trl 1.x dropped TrainingArguments.group_by_length,
       so we inject the sampler directly. On gfx1151 this is the single biggest lever measured: random
       batches mixing METAR (~460 tok) with TAF (~1100+ tok) pad every row to the longest, and 49% of
       linear compute was padding. Length-grouped batches bring that to ~1% (≈2x less linear work,
       ≈3x less attention work).
    2. Length-adaptive gradient checkpointing (`ckpt_above` tokens): recompute is a compute-for-memory
       trade, and on this 123 GiB unified-memory box it is only *needed* for the long tail. Measured
       (S24): no-checkpointing is 1.7x faster on the short half of the data but does NOT fit the
       ~2048-token TAF bucket at batch 6 (OOM, or worse: a swap storm that ends in a GPU page fault).
       HF checkpointing is a runtime flag checked per forward, so we flip it per microbatch on the
       padded length. With grouped batches lengths descend inside each megabatch, so the flag changes
       ~twice per megabatch (two cached compiled graphs under torch.compile)."""

    group_by_length = True
    ckpt_above = 0
    log_mem = False
    _ckpt_on = None

    def _get_train_sampler(self, train_dataset=None):
        ds = train_dataset if train_dataset is not None else self.train_dataset
        if not self.group_by_length or ds is None or "input_ids" not in ds.column_names:
            return super()._get_train_sampler(train_dataset)
        lengths = [len(x) for x in ds["input_ids"]]
        gen = torch.Generator()
        gen.manual_seed(self.args.seed)
        return LengthGroupedSampler(
            self.args.per_device_train_batch_size, lengths=lengths, generator=gen
        )

    def training_step(self, model, inputs, *args, **kwargs):
        if self.ckpt_above:
            want = inputs["input_ids"].shape[1] > self.ckpt_above
            if want != self._ckpt_on:
                m = model
                while hasattr(m, "_orig_mod"):  # torch.compile wrapper -> the PeftModel underneath
                    m = m._orig_mod
                (m.gradient_checkpointing_enable if want else m.gradient_checkpointing_disable)()
                self._ckpt_on = want
        out = super().training_step(model, inputs, *args, **kwargs)
        if self.log_mem:  # per-microbatch memory curve: how the compiled no-ckpt peak scales with length
            print(f"MEMLOG len={inputs['input_ids'].shape[1]} ckpt={self._ckpt_on} "
                  f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB "
                  f"reserved={torch.cuda.memory_reserved() / 2**30:.1f}GiB", flush=True)
            torch.cuda.reset_peak_memory_stats()
        return out


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
    # Resilience: a crash mid-run (S24's "GPU page fault" turned out to be a memory blow-up that
    # thrashed the unified pool; power events and driver hiccups remain) kills the process, so long
    # runs checkpoint every --save-steps and --resume picks up the latest checkpoint;
    # run_resilient.sh loops the two so a crash costs minutes, not the night.
    ap.add_argument("--save-steps", type=int, default=0, help="checkpoint every N steps (0=off)")
    ap.add_argument("--resume", action="store_true", help="resume from latest checkpoint in --out")
    # S24 memory levers. --ckpt-above N: checkpoint only microbatches longer than N tokens (see
    # AvtextSFTTrainer); implies checkpointing machinery on. --mem-fraction: on an APU the "GPU" pool
    # is host RAM, so an over-allocation is not a clean OOM but a swap storm that ends in an amdgpu
    # page fault and a hung box — a cap turns it back into a fast, retryable OOM (0 = off).
    ap.add_argument("--ckpt-above", type=int, default=0, help="adaptive checkpointing threshold (tokens)")
    ap.add_argument("--mem-fraction", type=float, default=0.85, help="cap on the GPU pool (0=off)")
    ap.add_argument("--log-mem", action="store_true", help="print peak GiB per microbatch (MEMLOG lines)")
    a = ap.parse_args()
    if a.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    if a.compile:  # variable seq lens -> automatic dynamic shapes; allow a few recompiles
        torch._dynamo.config.cache_size_limit = 64

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.config.use_cache = False
    if a.grad_checkpointing or a.ckpt_above:
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
        dataset_num_proc=1, bf16=True, gradient_checkpointing=a.grad_checkpointing and not a.ckpt_above,
        torch_compile=a.compile,
        save_strategy="steps" if a.save_steps else "no", save_steps=a.save_steps or 500,
        save_total_limit=2,
    )
    trainer = AvtextSFTTrainer(model=model, train_dataset=ds, args=cfg, processing_class=tok)
    trainer.group_by_length = a.group_by_length
    trainer.ckpt_above = a.ckpt_above
    trainer.log_mem = a.log_mem
    last = get_last_checkpoint(a.out) if (a.resume and os.path.isdir(a.out)) else None
    if last:
        print("RESUMING_FROM", last, flush=True)
    trainer.train(resume_from_checkpoint=last)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("SAVED_ADAPTER", a.out)


if __name__ == "__main__":
    main()
