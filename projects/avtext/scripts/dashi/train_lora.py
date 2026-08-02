"""Unsloth bf16 LoRA finetune for the aviation-decode task (Phase 4, dashi/gfx1151)."""
import argparse, os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.expanduser("~/ft/train.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--packing", action="store_true",
                    help="pack short examples into max_seq sequences (much faster on mixed-length sets)")
    a = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.model, max_seq_length=a.max_seq, dtype=torch.bfloat16, load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.0, bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42)

    ds = load_dataset("json", data_files=a.data, split="train")
    ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)},
                remove_columns=ds.column_names)

    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text", packing=a.packing,
            per_device_train_batch_size=a.batch, gradient_accumulation_steps=2,
            warmup_steps=10, num_train_epochs=a.epochs, max_steps=a.max_steps,
            learning_rate=a.lr, logging_steps=5, optim="adamw_torch", weight_decay=0.01,
            lr_scheduler_type="linear", seed=42, output_dir=a.out, report_to="none",
            max_seq_length=a.max_seq, dataset_num_proc=1, bf16=True))
    trainer.train()
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print("SAVED_ADAPTER", a.out)

if __name__ == "__main__":
    main()
