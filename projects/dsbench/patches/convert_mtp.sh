#!/bin/bash
# Export the Qwen3.6-35B-A3B MTP head to GGUF with this fork's converter (`--mtp`: the nextn layer
# plus the shared embeddings, final norm and lm_head), for the MTP-acceptance leg of the battery.
# Source: only shards 1, 25 and 26 of the HF checkpoint (fetch_mtp_src.py). The converter opens
# every shard the index names, so the index is narrowed to those three (the full one is kept as
# model.safetensors.index.full.json). Runs in the ROCm toolbox with the ftgguf venv (torch).
set -eu
SRC=~/benchlab/scratch/qwen36-mtp-src
OUT=~/models/Qwen3.6-35B-A3B-MTP-GGUF/Q8_0/mtp-Qwen3.6-35B-A3B-Q8_0.gguf
mkdir -p "$(dirname "$OUT")"
cd "$SRC"
[ -f model.safetensors.index.full.json ] || mv model.safetensors.index.json model.safetensors.index.full.json
python3 - <<'PY'
import json
full = json.load(open("model.safetensors.index.full.json"))
have = {"model-00001-of-00026.safetensors", "model-00025-of-00026.safetensors",
        "model-00026-of-00026.safetensors"}
full["weight_map"] = {k: v for k, v in full["weight_map"].items() if v in have}
json.dump(full, open("model.safetensors.index.json", "w"), indent=1)
print(len(full["weight_map"]), "tensors indexed from", len(have), "shards")
PY
toolbox run --container llama-rocm-unlimited-build bash -c "source ~/ftgguf/bin/activate && \
  cd ~/src/llama-qwen4exp-src && PYTHONPATH=gguf-py python convert_hf_to_gguf.py $SRC --mtp \
  --outtype q8_0 --outfile $OUT" </dev/null
ls -la "$OUT"
