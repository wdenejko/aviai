#!/usr/bin/env bash
# phase_all.sh — all-products (METAR+TAF+NOTAM) rank-16 finetune on dashi (Session 22 / Phase 7).
#
# The full multi-task bet: ONE rank-16 LoRA on train_all.jsonl (~29k: 8k METAR + 8k TAF + 9k NOTAM
# extraction + 4k NOTAM classification, interleaved), evaluated on ALL FOUR frozen evals with the
# SAME adapter. Question: does one adapter cover three product families without regressing any vs
# the single-task / two-task adapters? Base numbers for METAR/TAF already exist; NOTAM base comes
# from phase_notam — so this run only needs the FT side of the four-way comparison.
# Same conventions as phase_combined.sh (train = ~/ft unsloth env; harness = uv venv; serial).
set -uo pipefail

BASE=~/models/gemma-4/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf
ADAPTER_DIR=~/ft/gemma-4/gemma-4-e4b-all-r16
ADAPTER_GGUF=~/ft/gemma-4/gemma-4-e4b-all-r16-f16.gguf
DATA=~/ft/train_all.jsonl
REPO=~/avtext/projects/avtext
SERVE=~/scripts/avtext/serve.sh
LIMIT=${LIMIT:-1500}   # same subset as every other run -> directly comparable
BATCH=6
MAXSEQ=2048            # must fit the long TAF targets; METAR/NOTAM examples are short

stop8080 () {
  p=$(ss -tlnHp 2>/dev/null | grep ':8080 ' | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "${p:-}" ] && { kill "$p" 2>/dev/null; for i in $(seq 1 20); do ss -tlnH | grep -q ':8080 ' || break; sleep 1; done; }
}

echo "=== PHASE_ALL START $(date) | limit=$LIMIT batch=$BATCH max_seq=$MAXSEQ ==="
stop8080

# 1. Try packing (packs the many short METAR/NOTAM examples instead of padding each to 2048). It was
# accepted-but-no-speedup on this ROCm build before, so dry-run 5 steps and fall back to non-packing
# (correct, just slow) on any error. Same data/experiment either way.
echo "=== PACKING DRY-RUN $(date +%H:%M) ==="
PACK=""
HF_HUB_DISABLE_XET=1 ~/ft/bin/python ~/scripts/avtext/train_lora.py \
  --model unsloth/gemma-4-E4B-it --out /tmp/pack_test --data "$DATA" \
  --rank 16 --batch "$BATCH" --max-seq "$MAXSEQ" --packing --max-steps 5 \
  > ~/logs/pack_test_all.log 2>&1
if grep -q SAVED_ADAPTER ~/logs/pack_test_all.log; then
  PACK="--packing"; echo "packing OK -> real run uses --packing"
else
  echo "packing dry-run FAILED -> non-packing fallback (slow)"; tail -4 ~/logs/pack_test_all.log
fi
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

# 2. Serve the ONE all-products adapter; eval it on all four products with the same served model.
echo "=== SERVE all $(date +%H:%M) ==="
LORA="$ADAPTER_GGUF" "$SERVE" "$BASE" gemma-4-e4b-all-r16 || echo "serve all FAILED"

echo "=== EVAL all on METAR $(date +%H:%M) ==="
( cd "$REPO" && uv run python -m avtext.harness.runner --completion \
    --base-url http://127.0.0.1:8080 --eval eval/v2/eval.jsonl --limit "$LIMIT" \
    --out-dir reports/gemma-4/runs \
    --model gemma-4-e4b-all-r16-metar --model-id gemma-4-e4b-all-r16-metar ) || echo "METAR eval FAILED"

echo "=== EVAL all on TAF $(date +%H:%M) ==="
( cd "$REPO" && uv run python -m avtext.harness.runner_taf --completion \
    --base-url http://127.0.0.1:8080 --eval eval/taf/v1/eval.jsonl --limit "$LIMIT" \
    --max-tokens 1024 --out-dir reports/gemma-4/runs \
    --model gemma-4-e4b-all-r16-taf --model-id gemma-4-e4b-all-r16-taf ) || echo "TAF eval FAILED"

echo "=== EVAL all on NOTAM-extraction $(date +%H:%M) ==="
( cd "$REPO" && uv run python -m avtext.harness.runner_notam \
    --base-url http://127.0.0.1:8080 --eval eval/notam/v1/eval.jsonl --limit "$LIMIT" \
    --out-dir reports/gemma-4/runs \
    --model gemma-4-e4b-all-r16-notam-ext --model-id gemma-4-e4b-all-r16-notam-ext ) || echo "NOTAM-ext eval FAILED"

echo "=== EVAL all on NOTAM-classification $(date +%H:%M) ==="
( cd "$REPO" && uv run python -m avtext.harness.runner_notam \
    --base-url http://127.0.0.1:8080 --eval eval/notam_cls/v1/eval.jsonl --limit "$LIMIT" \
    --out-dir reports/gemma-4/runs \
    --model gemma-4-e4b-all-r16-notam-cls --model-id gemma-4-e4b-all-r16-notam-cls ) || echo "NOTAM-cls eval FAILED"
stop8080

# 3. RESULTS (decode runs carry 'overall'; the classification run carries 'metrics')
echo "=== RESULTS $(date +%H:%M) ==="
cd "$REPO"
for id in all-r16-metar all-r16-taf all-r16-notam-ext all-r16-notam-cls; do
  d=$(ls -dt reports/gemma-4/runs/*"$id"* 2>/dev/null | head -1)
  [ -n "$d" ] && python3 -c "
import json; r=json.load(open('$d/run.json'))
if 'overall' in r:
    o=r['overall']; print(f\"$id: value={o['recall'] or 0:.1%} halluc={o['hallucination_rate'] or 0:.1%} EM={o['exact_match']:.1%} invalid={r.get('n_invalid')}\")
else:
    m=r['metrics']; print(f\"$id: accuracy={m['accuracy']:.1%} macro_f1={m['macro_f1']:.1%} invalid={m['invalid']}\")"
done
echo "=== PHASE_ALL DONE $(date) ==="
