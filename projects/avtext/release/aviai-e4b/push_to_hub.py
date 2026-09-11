"""Upload the aviai-e4b release to the Hugging Face Hub — run by the maintainer with their
own token, on the machine that holds the weights (dashi: ~/fttrain/release).

    uv run python push_to_hub.py --repo-id <namespace>/aviai-e4b \\
        --weights ~/fttrain/release/aviai-e4b \\
        --gguf ~/fttrain/release/aviai-e4b-Q8_0.gguf \\
               ~/fttrain/release/aviai-e4b-f16.gguf \\
        --adapter ~/fttrain/adapter-all-lg --adapter-gguf ~/fttrain/adapter-all-lg-f16.gguf \\
        [--private] [--dry-run]

Authenticate first with `hf auth login` (or set HF_TOKEN). Nothing here reads or stores the token.
Layout in the repo: merged Transformers checkpoint at the root, GGUFs at the root, the original LoRA
under adapter/, prompt templates under prompts/, plus README / NOTICE / LICENSE from this directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--weights", required=True, help="merged Transformers checkpoint directory")
    ap.add_argument("--gguf", nargs="*", default=[], help="merged GGUF file(s)")
    ap.add_argument(
        "--adapter", default=None, help="PEFT adapter directory (uploaded under adapter/)"
    )
    ap.add_argument("--adapter-gguf", default=None, help="GGUF LoRA file (uploaded under adapter/)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    w = Path(a.weights).expanduser()
    for f in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        if not (w / f).is_file():
            raise SystemExit(f"{w} is not a complete merged checkpoint: missing {f}")
    plan: list[tuple[str, str]] = [(str(w), "<folder> -> /")]
    plan += [(g, Path(g).name) for g in a.gguf]
    if a.adapter:
        plan.append((a.adapter, "<folder> -> adapter/"))
    if a.adapter_gguf:
        plan.append((a.adapter_gguf, "adapter/" + Path(a.adapter_gguf).name))
    plan += [(str(HERE / f), f) for f in ("README.md", "NOTICE", "LICENSE")]
    plan.append((str(HERE / "prompts"), "<folder> -> prompts/"))
    for src, dst in plan:
        print(f"  {src}  ->  {dst}")
    if a.dry_run:
        print("(dry run)")
        return
    api = HfApi()
    api.create_repo(a.repo_id, repo_type="model", private=a.private, exist_ok=True)
    api.upload_folder(folder_path=str(w), repo_id=a.repo_id, path_in_repo=".", repo_type="model")
    for g in a.gguf:
        api.upload_file(path_or_fileobj=g, path_in_repo=Path(g).name, repo_id=a.repo_id)
    if a.adapter:
        api.upload_folder(
            folder_path=a.adapter, repo_id=a.repo_id, path_in_repo="adapter", repo_type="model",
            ignore_patterns=["README.md", "checkpoint-*/**"],
        )  # fmt: skip
    if a.adapter_gguf:
        api.upload_file(
            path_or_fileobj=a.adapter_gguf,
            path_in_repo="adapter/" + Path(a.adapter_gguf).name,
            repo_id=a.repo_id,
        )
    for f in ("README.md", "NOTICE", "LICENSE"):
        api.upload_file(path_or_fileobj=str(HERE / f), path_in_repo=f, repo_id=a.repo_id)
    api.upload_folder(folder_path=str(HERE / "prompts"), repo_id=a.repo_id, path_in_repo="prompts")
    print(f"uploaded -> https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
