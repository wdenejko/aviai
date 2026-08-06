#!/usr/bin/env bash
# eval_notam_chunked.sh LABEL [LORA_GGUF] — chunked NOTAM extraction + full classification.
#
# Fallback for the serving-degradation bug (llama-server rots into <unused49> spam after ~100
# heavy-generation records on this ROCm build). Extraction is run in small chunks with a FULL
# SERVER RESTART between chunks, so cumulative KV/memory state never builds up; merge_notam then
# stitches the chunks into one normal run.json. Classification (1-token outputs, stable) runs full
# in a single serve. Env: CHUNK (default 100), EXTTOK (512), FLASH (off).
#   usage:  bash eval_notam_chunked.sh gemma-4-e4b-notam-r16 ~/ft/gemma-4/gemma-4-e4b-notam-r16-f16.gguf
#           bash eval_notam_chunked.sh gemma-4-e4b-notam-base            # base = no LoRA
set -uo pipefail
LABEL="$1"; LORA_GGUF="${2:-}"
BASE=~/models/gemma-4/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf
REPO=~/avtext/projects/avtext; TB=llama-rocm-7.2.4_2
EXT=eval/notam/v1/eval.jsonl; CLS=eval/notam_cls/v1/eval.jsonl
CHUNK="${CHUNK:-100}"; EXTTOK="${EXTTOK:-512}"; FLASH="${FLASH:-off}"
CHUNKDIR=reports/gemma-4/runs/chunks/${LABEL}-ext

stop8080 () {
  p=$(ss -tlnHp 2>/dev/null | grep ':8080 ' | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "${p:-}" ] && { kill "$p" 2>/dev/null; for i in $(seq 1 20); do ss -tlnH | grep -q ':8080 ' || break; sleep 1; done; }
}
serve () {
  local la=""; [ -n "$LORA_GGUF" ] && la="--lora $LORA_GGUF"
  nohup toolbox run --container "$TB" llama-server --model "$BASE" --alias "$LABEL" $la \
    --host 0.0.0.0 --port 8080 --n-gpu-layers all --ctx-size 8192 --parallel 1 \
    --flash-attn "$FLASH" --jinja --reasoning-budget 0 --no-webui >~/srv-8080.log 2>&1 </dev/null &
  for i in $(seq 1 90); do curl -s -m 3 localhost:8080/health 2>/dev/null | grep -q ok && return 0; sleep 2; done
  echo "SERVE FAILED for $LABEL"; return 1
}

cd "$REPO"
NREC=$(wc -l < "$EXT")
echo "=== CHUNKED $LABEL START $(date) | NREC=$NREC CHUNK=$CHUNK EXTTOK=$EXTTOK FLASH=$FLASH ==="
rm -rf "$CHUNKDIR"
off=0; ci=0
while [ "$off" -lt "$NREC" ]; do
  stop8080; serve || echo "serve fail chunk $ci"
  echo "--- ext chunk $ci offset=$off $(date +%H:%M) ---"
  uv run python -m avtext.harness.runner_notam --base-url http://127.0.0.1:8080 \
    --eval "$EXT" --offset "$off" --limit "$CHUNK" --max-tokens "$EXTTOK" \
    --out-dir "$CHUNKDIR" --out-name "$(printf 'c%02d' "$ci")" \
    --model "$LABEL-ext" --model-id "$LABEL-ext" || echo "chunk $ci FAILED"
  off=$((off + CHUNK)); ci=$((ci + 1))
done
stop8080
echo "=== MERGE ext $(date +%H:%M) ==="
uv run python -m avtext.report.merge_notam --chunks "$CHUNKDIR" \
  --out-dir reports/gemma-4/runs --model-id "$LABEL-ext" || echo "MERGE FAILED"

echo "=== CLS full $(date +%H:%M) ==="
stop8080; serve || echo "serve fail cls"
uv run python -m avtext.harness.runner_notam --base-url http://127.0.0.1:8080 \
  --eval "$CLS" --out-dir reports/gemma-4/runs \
  --model "$LABEL-cls" --model-id "$LABEL-cls" || echo "cls FAILED"
stop8080
echo "=== CHUNKED $LABEL DONE $(date) ==="
