"""Upload this release package to the Hugging Face Hub — run by the maintainer with their own token.

    uv run python release/gemma-4-e4b-avtext-lora-r16/push_to_hub.py --repo-id <namespace>/gemma-4-e4b-avtext-lora-r16 [--private]

Authenticate first with `hf auth login` (or set HF_TOKEN in the environment). Nothing here reads or
stores the token; huggingface_hub picks it up from its own cache/env. Adapter files go to the repo
root (so `PeftModel.from_pretrained(<repo id>)` works), the GGUF LoRA alongside, plus README/NOTICE/
LICENSE and the prompt templates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"
REQUIRED = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded")
    a = ap.parse_args()
    missing = [f for f in REQUIRED if not (WEIGHTS / f).is_file()]
    if missing:
        raise SystemExit(f"missing adapter files in {WEIGHTS}: {missing} (pull them from dashi ~/fttrain/adapter-all-lg)")
    if "GRANT REFERENCE: _to be filled" in (HERE / "README.md").read_text(encoding="utf-8"):
        raise SystemExit("README.md still has the OpenNOTAM grant placeholder — fill it in before publishing")
    files = sorted(p for p in WEIGHTS.iterdir() if p.is_file() and p.name != "README.md")
    files += [HERE / "README.md", HERE / "NOTICE", HERE / "LICENSE"]
    files += sorted((HERE / "prompts").iterdir())
    for p in files:
        print(f"  {p.relative_to(HERE)}  ->  {p.name if p.parent != HERE / 'prompts' else 'prompts/' + p.name}")
    if a.dry_run:
        return
    api = HfApi()
    api.create_repo(a.repo_id, repo_type="model", private=a.private, exist_ok=True)
    for p in files:
        dest = ("prompts/" + p.name) if p.parent == HERE / "prompts" else p.name
        api.upload_file(path_or_fileobj=str(p), path_in_repo=dest, repo_id=a.repo_id, repo_type="model")
    print(f"uploaded -> https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
