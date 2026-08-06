#!/usr/bin/env bash
# phase_all_v2.sh — all-products (METAR+TAF+NOTAM) rank-16 finetune, chunked NOTAM eval (Session 22).
#
# Same as phase_all.sh but routes the NOTAM evals through eval_notam_chunked.sh (chunk + server
# restart) to dodge the <unused49> serving-degradation bug; METAR/TAF stay single-serve (short
# outputs, stable). Trains ONE rank-16 adapter on train_all.jsonl (~29k) and evals it on all four.
set -uo pipefail

BASE=~/models/gemma-4/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf
ADAPTER_DIR=~/ft/gemma-4/gemma-4-e4b-all-r16
ADAPTER_GGUF=~/ft/gemma-4/gemma-4-e4b-all-r16-f16.gguf
DATA=~/ft/train_all.jsonl
REPO=~/avtext/projects/avtext
SERVE=~/scripts/avtext/serve.sh
LIMIT=${LIMIT:-1500}   # METAR/TAF subset (matched-N with the other runs)
BATCH=6; MAXSEQ=2048

stop8080 () {
  p=$(ss -tlnHp 2>/dev/null | grep ':8080 ' | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "${p:-}" ] && { kill "$p" 2>/dev/null; for i in $(seq 1 20); do ss -tlnH | grep -q ':8080 ' || break; sleep 1; done; }
}

echo "=== PHASE_ALL_V2 START $(date) | limit=$LIMIT batch=$BATCH max_seq=$MAXSEQ ==="
stop8080

# 1. Packing dry-run (accepted-but-no-speedup before; try, fall back to non-packing on any error)
echo "=== PACKING DRY-RUN $(date +%H:%M) ==="
PACK=""
HF_HUB_DISABLE_XET=1 ~/ft/bin/python ~/scripts/avtext/train_lora.py \
  --model unsloth/gemma-4-E4B-it --out /tmp/pack_test --data "$DATA" \
  --rank 16 --batch "$BATCH" --max-seq "$MAXSEQ" --packing --max-steps 5 \
  > ~/logs/pack_test_all.log 2>&1
if grep -q SAVED_ADAPTER ~/logs/pack_test_all.log; then PACK="--packing"; echo "packing OK"; else
  echo "packing FAILED -> non-packing"; tail -4 ~/logs/pack_test_all.log; fi
rm -rf /tmp/pack_test

echo "=== TRAIN all r16 $(date +%H:%M) | pack='${PACK:-none}' ==="
HF_HUB_DISABLE_XET=1 ~/ft/bin/python ~/scripts/avtext/train_lora.py \
  --model unsloth/gemma-4-E4B-it --out "$ADAPTER_DIR" --data "$DATA" \
  --rank 16 --epochs 1 --batch "$BATCH" --max-seq "$MAXSEQ" $PACK \
  2>&1 | grep -iE 'trainable param|train_loss|SAVED_ADAPTER|error|traceback|out of memory' | tail -6

echo "=== CONVERT $(date +%H:%M) ==="
[ -f /tmp/llamacpp-convert/convert_lora_to_gguf.py ] || \
  (cd /tmp && git clone --depth 1 https://github.com/ggml-org/llama.cpp llamacpp-convert >/dev/null 2>&1)
HFBASE=$(ls -d ~/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it/snapshots/*/ | head -1)
PYTHONPATH=/tmp/llamacpp-convert/gguf-py ~/ft/bin/python /tmp/llamacpp-convert/convert_lora_to_gguf.py \
  "$ADAPTER_DIR" --base "$HFBASE" --outfile "$ADAPTER_GGUF" --outtype f16 2>&1 | tail -1

# 2. NOTAM evals (chunked + restart, both tasks) for the all-products adapter
echo "=== NOTAM (chunked) all-r16 $(date +%H:%M) ==="
bash ~/scripts/avtext/eval_notam_chunked.sh gemma-4-e4b-all-r16 "$ADAPTER_GGUF"

# 3. METAR + TAF (single serve; short outputs, stable)
echo "=== SERVE all-r16 for METAR/TAF $(date +%H:%M) ==="
LORA="$ADAPTER_GGUF" "$SERVE" "$BASE" gemma-4-e4b-all-r16 || echo "serve FAILED"
echo "=== EVAL METAR $(date +%H:%M) ==="
( cd "$REPO" && uv run python -m avtext.harness.runner --completion \
    --base-url http://127.0.0.1:8080 --eval eval/v2/eval.jsonl --limit "$LIMIT" \
    --out-dir reports/gemma-4/runs \
    --model gemma-4-e4b-all-r16-metar --model-id gemma-4-e4b-all-r16-metar ) || echo "METAR eval FAILED"
echo "=== EVAL TAF $(date +%H:%M) ==="
( cd "$REPO" && uv run python -m avtext.harness.runner_taf --completion \
    --base-url http://127.0.0.1:8080 --eval eval/taf/v1/eval.jsonl --limit "$LIMIT" \
    --max-tokens 1024 --out-dir reports/gemma-4/runs \
    --model gemma-4-e4b-all-r16-taf --model-id gemma-4-e4b-all-r16-taf ) || echo "TAF eval FAILED"
stop8080
echo "=== PHASE_ALL_V2 DONE $(date) ==="
