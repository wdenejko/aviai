#!/usr/bin/env bash
# phase_combined.sh — multi-task METAR+TAF finetune on dashi (Session 20 / Phase 6).
#
# Trains ONE rank-16 LoRA on train_combined.jsonl (8k METAR + 8k TAF, interleaved) and evals the
# SAME adapter on BOTH frozen evals: TAF via runner_taf, METAR via runner. The question: does one
# adapter do both without regressing either vs the single-task adapters? Serial (no GPU contention);
# same conventions as phase_taf.sh (train = ~/ft unsloth env; harness = uv venv hitting :8080).
set -uo pipefail

BASE=~/models/gemma-4/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf
ADAPTER_DIR=~/ft/gemma-4/gemma-4-e4b-combined-r16
ADAPTER_GGUF=~/ft/gemma-4/gemma-4-e4b-combined-r16-f16.gguf
DATA=~/ft/train_combined.jsonl
REPO=~/avtext/projects/avtext
SERVE=~/scripts/avtext/serve.sh
LIMIT=${LIMIT:-1500}   # same subset as the single-task runs -> directly comparable
BATCH=6
MAXSEQ=2048            # must fit the long TAF targets; METAR examples are short

stop8080 () {
  p=$(ss -tlnHp 2>/dev/null | grep ':8080 ' | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "${p:-}" ] && { kill "$p" 2>/dev/null; for i in $(seq 1 20); do ss -tlnH | grep -q ':8080 ' || break; sleep 1; done; }
}

echo "=== PHASE_COMBINED START $(date) | limit=$LIMIT batch=$BATCH max_seq=$MAXSEQ ==="
stop8080

# 1. TRAIN one rank-16 adapter on the combined set + convert to GGUF
echo "=== TRAIN combined r16 $(date +%H:%M) ==="
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

# 2. Serve the ONE combined adapter, eval it on BOTH products
echo "=== SERVE combined $(date +%H:%M) ==="
LORA="$ADAPTER_GGUF" "$SERVE" "$BASE" gemma-4-e4b-combined-r16 || echo "serve combined FAILED"

echo "=== EVAL combined on TAF $(date +%H:%M) ==="
( cd "$REPO" && uv run python -m avtext.harness.runner_taf --completion \
    --base-url http://127.0.0.1:8080 --eval eval/taf/v1/eval.jsonl --limit "$LIMIT" \
    --max-tokens 1024 --out-dir reports/gemma-4/runs \
    --model gemma-4-e4b-combined-r16-taf --model-id gemma-4-e4b-combined-r16-taf ) || echo "TAF eval FAILED"

echo "=== EVAL combined on METAR $(date +%H:%M) ==="
( cd "$REPO" && uv run python -m avtext.harness.runner --completion \
    --base-url http://127.0.0.1:8080 --eval eval/v2/eval.jsonl --limit "$LIMIT" \
    --out-dir reports/gemma-4/runs \
    --model gemma-4-e4b-combined-r16-metar --model-id gemma-4-e4b-combined-r16-metar ) || echo "METAR eval FAILED"
stop8080

# 3. RESULTS
echo "=== RESULTS $(date +%H:%M) ==="
cd "$REPO"
for id in combined-r16-taf combined-r16-metar; do
  d=$(ls -dt reports/gemma-4/runs/*"$id"* 2>/dev/null | head -1)
  [ -n "$d" ] && python3 -c "
import json; r=json.load(open('$d/run.json')); o=r['overall']
print(f\"$id: value={o['recall']:.1%} halluc={o['hallucination_rate']:.1%} EM={o['exact_match']:.1%} invalid={r['n_invalid']}\")"
done
echo "=== PHASE_COMBINED DONE $(date) ==="
