#!/usr/bin/env bash
# phase_taf.sh — unattended TAF measurement on dashi (Session 20 / Phase 6).
#
# Extends phase_d1.sh: TRAIN rank-16 -> convert LoRA->GGUF -> BASELINE eval -> FINETUNED eval.
# Order is deliberate: training (the must-happen GPU work) runs FIRST, so a slow/failed eval never
# costs us the adapter. Everything is serial — the harness hits the served model on :8080, and
# training also needs the GPU, so they must never overlap.
#   - training uses ~/ft/bin/python (unsloth/ROCm env) + train_lora.py
#   - the harness uses the uv venv at ~/avtext (avtext pkg + deps), hitting 127.0.0.1:8080
#   - serving uses serve.sh (llama.cpp toolbox; --parallel 1, --reasoning-budget 0)
set -uo pipefail

BASE=~/models/gemma-4/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf
ADAPTER_DIR=~/ft/gemma-4/gemma-4-e4b-taf-r16
ADAPTER_GGUF=~/ft/gemma-4/gemma-4-e4b-taf-r16-f16.gguf
DATA=~/ft/train_taf.jsonl
REPO=~/avtext/projects/avtext
SERVE=~/scripts/avtext/serve.sh
LIMIT=${LIMIT:-1500}   # eval subset (base+finetuned scored on the SAME first-N records, comparable)

# batch 6 x max-seq 2048 = 12,288 tokens/batch == the proven v2 run's footprint (16 x 768), but long
# enough for TAF targets (p99 ~1,430 tok); shorter would truncate the valuable multi-period forecasts.
BATCH=6
MAXSEQ=2048

runner () {  # $1 = model alias/id
  ( cd "$REPO" && uv run python -m avtext.harness.runner_taf --completion \
      --base-url http://127.0.0.1:8080 --eval eval/taf/v1/eval.jsonl \
      --limit "$LIMIT" --max-tokens 1024 --out-dir reports/gemma-4/runs \
      --model "$1" --model-id "$1" ) || echo "EVAL FAILED for $1"
}
stop8080 () {
  p=$(ss -tlnHp 2>/dev/null | grep ':8080 ' | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "${p:-}" ] && { kill "$p" 2>/dev/null; for i in $(seq 1 20); do ss -tlnH | grep -q ':8080 ' || break; sleep 1; done; }
}

echo "=== PHASE_TAF START $(date) | limit=$LIMIT batch=$BATCH max_seq=$MAXSEQ ==="
stop8080  # free the GPU

# 1. TRAIN rank-16 (front-loaded) + convert to GGUF
echo "=== TRAIN r16 $(date +%H:%M) ==="
HF_HUB_DISABLE_XET=1 ~/ft/bin/python ~/scripts/avtext/train_lora.py \
  --model unsloth/gemma-4-E4B-it --out "$ADAPTER_DIR" --data "$DATA" \
  --rank 16 --epochs 1 --batch "$BATCH" --max-seq "$MAXSEQ" \
  2>&1 | grep -iE 'trainable param|train_loss|SAVED_ADAPTER|error|traceback|out of memory' | tail -6

echo "=== CONVERT $(date +%H:%M) ==="
[ -f /tmp/llamacpp-convert/convert_lora_to_gguf.py ] || \
  (cd /tmp && git clone --depth 1 https://github.com/ggml-org/llama.cpp llamacpp-convert >/dev/null 2>&1)
HFBASE=$(ls -d ~/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it/snapshots/*/ | head -1)
PYTHONPATH=/tmp/llamacpp-convert/gguf-py ~/ft/bin/python /tmp/llamacpp-convert/convert_lora_to_gguf.py \
  "$ADAPTER_DIR" --base "$HFBASE" --outfile "$ADAPTER_GGUF" --outtype f16 2>&1 | tail -1

# 2. FINETUNED eval FIRST (the priority result — secure it before the base model's slower eval)
echo "=== SERVE finetuned $(date +%H:%M) ==="
LORA="$ADAPTER_GGUF" "$SERVE" "$BASE" gemma-4-e4b-taf-r16 || echo "serve finetuned FAILED"
echo "=== FINETUNED EVAL $(date +%H:%M) ==="
runner gemma-4-e4b-taf-r16
stop8080

# 3. BASELINE eval (base model; slower — it tends to ramble to the token cap)
echo "=== SERVE base $(date +%H:%M) ==="
"$SERVE" "$BASE" gemma-4-e4b-taf-base || echo "serve base FAILED"
echo "=== BASELINE EVAL $(date +%H:%M) ==="
runner gemma-4-e4b-taf-base
stop8080

# 4. RESULTS
echo "=== RESULTS $(date +%H:%M) ==="
cd "$REPO"
for id in taf-base taf-r16; do
  d=$(ls -dt reports/gemma-4/runs/*"$id"* 2>/dev/null | head -1)
  [ -n "$d" ] && python3 -c "
import json; r=json.load(open('$d/run.json')); o=r['overall']
print(f\"$id: value={o['recall']:.1%} halluc={o['hallucination_rate']:.1%} EM={o['exact_match']:.1%} \"
      f\"period-match={r['n_period_match']}/{o['n_records']} invalid={r['n_invalid']}\")"
done
echo "=== PHASE_TAF DONE $(date) ==="
