#!/usr/bin/env bash
# phase_notam.sh — NOTAM-only rank-16 finetune on dashi (Session 22 / Phase 7).
#
# Trains ONE rank-16 LoRA on train_notam.jsonl (extraction + classification interleaved) and evals
# it on BOTH NOTAM evals via runner_notam (auto-detects task from the manifest kind). Captures the
# FULL base-vs-FT comparison: base extraction/classification are evaluated too, so the dashboard has
# every cell. Same conventions as phase_taf/phase_combined (train = ~/ft unsloth env; harness = uv
# venv hitting :8080; serial — never overlap training and serving on the one iGPU).
#   NOTE: runner_notam is completion-only (no --completion flag); it POSTs to /completion directly.
set -uo pipefail

BASE=~/models/gemma-4/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf
ADAPTER_DIR=~/ft/gemma-4/gemma-4-e4b-notam-r16
ADAPTER_GGUF=~/ft/gemma-4/gemma-4-e4b-notam-r16-f16.gguf
DATA=~/ft/train_notam.jsonl
REPO=~/avtext/projects/avtext
SERVE=~/scripts/avtext/serve.sh
LIMIT=${LIMIT:-1500}   # eval subset (base+FT scored on the SAME first-N records -> comparable)
BATCH=6
MAXSEQ=2048            # NOTAM targets are short, but keep the proven footprint for a clean compare

EXT=eval/notam/v1/eval.jsonl
CLS=eval/notam_cls/v1/eval.jsonl

notam_ext () {  # $1 = model id
  ( cd "$REPO" && uv run python -m avtext.harness.runner_notam \
      --base-url http://127.0.0.1:8080 --eval "$EXT" --limit "$LIMIT" \
      --out-dir reports/gemma-4/runs --model "$1" --model-id "$1" ) || echo "EXT eval FAILED for $1"
}
notam_cls () {  # $1 = model id
  ( cd "$REPO" && uv run python -m avtext.harness.runner_notam \
      --base-url http://127.0.0.1:8080 --eval "$CLS" --limit "$LIMIT" \
      --out-dir reports/gemma-4/runs --model "$1" --model-id "$1" ) || echo "CLS eval FAILED for $1"
}
stop8080 () {
  p=$(ss -tlnHp 2>/dev/null | grep ':8080 ' | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "${p:-}" ] && { kill "$p" 2>/dev/null; for i in $(seq 1 20); do ss -tlnH | grep -q ':8080 ' || break; sleep 1; done; }
}

echo "=== PHASE_NOTAM START $(date) | limit=$LIMIT batch=$BATCH max_seq=$MAXSEQ ==="
stop8080  # free the GPU

# 1. TRAIN rank-16 (front-loaded — secure the adapter before any slow eval) + convert to GGUF
echo "=== TRAIN notam r16 $(date +%H:%M) ==="
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

# 2. FINETUNED eval FIRST (the priority result — both NOTAM tasks on the one adapter)
echo "=== SERVE finetuned $(date +%H:%M) ==="
LORA="$ADAPTER_GGUF" "$SERVE" "$BASE" gemma-4-e4b-notam-r16 || echo "serve finetuned FAILED"
echo "=== FT EXT EVAL $(date +%H:%M) ===";  notam_ext gemma-4-e4b-notam-r16-ext
echo "=== FT CLS EVAL $(date +%H:%M) ===";  notam_cls gemma-4-e4b-notam-r16-cls
stop8080

# 3. BASELINE eval (base model; the before-picture for both tasks)
echo "=== SERVE base $(date +%H:%M) ==="
"$SERVE" "$BASE" gemma-4-e4b-notam-base || echo "serve base FAILED"
echo "=== BASE EXT EVAL $(date +%H:%M) ===";  notam_ext gemma-4-e4b-notam-base-ext
echo "=== BASE CLS EVAL $(date +%H:%M) ===";  notam_cls gemma-4-e4b-notam-base-cls
stop8080

# 4. RESULTS (extraction has run.json['overall']; classification has run.json['metrics'])
echo "=== RESULTS $(date +%H:%M) ==="
cd "$REPO"
for id in notam-base-ext notam-r16-ext notam-base-cls notam-r16-cls; do
  d=$(ls -dt reports/gemma-4/runs/*"$id"* 2>/dev/null | head -1)
  [ -n "$d" ] && python3 -c "
import json; r=json.load(open('$d/run.json'))
if 'overall' in r:
    o=r['overall']; print(f\"$id: value={o['recall'] or 0:.1%} halluc={o['hallucination_rate'] or 0:.1%} EM={o['exact_match']:.1%} invalid={r.get('n_invalid')}\")
else:
    m=r['metrics']; print(f\"$id: accuracy={m['accuracy']:.1%} macro_f1={m['macro_f1']:.1%} invalid={m['invalid']}\")"
done
echo "=== PHASE_NOTAM DONE $(date) ==="
