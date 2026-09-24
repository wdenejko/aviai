"""Fetch only what the fork converter needs to export the Qwen3.6-35B-A3B MTP head (--mtp):
config/tokenizer files, the shard index, and shards 1 (embed_tokens), 25 and 26 (mtp.*, lm_head)."""
from huggingface_hub import hf_hub_download

REPO, REV = "Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"
DEST = "/home/wdenejko/benchlab/scratch/qwen36-mtp-src"
files = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
         "vocab.json", "merges.txt", "chat_template.jinja", "model.safetensors.index.json",
         "model-00001-of-00026.safetensors", "model-00025-of-00026.safetensors",
         "model-00026-of-00026.safetensors"]
for f in files:
    print(hf_hub_download(REPO, f, revision=REV, local_dir=DEST), flush=True)
