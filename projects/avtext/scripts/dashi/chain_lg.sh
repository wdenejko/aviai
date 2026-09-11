#!/usr/bin/env bash
# chain_lg.sh — evaluate the S23 grown all-products adapter (adapter-all-lg) across all four
# families on the SAME frozen evals as Phase 7, so "grown 49k vs Phase-7 29k" is a clean A/B.
#
# Order: convert LoRA->GGUF, then METAR (eval/v2) -> TAF (eval/taf/v1) -> NOTAM ext(chunked,
# grammar-on)+cls, each serve/eval/stop so training-vs-serving never overlap on the shared iGPU.
# Paths reflect the post-wipe dashi layout: base GGUF + adapter live in ~/fttrain/ (S23 rebuild).
set -uo pipefail
BASE="${BASE:-$HOME/fttrain/gemma-4-E4B-it-Q8_0.gguf}"
ADAPTER_DIR="${ADAPTER_DIR:-$HOME/fttrain/adapter-all-lg}"
ADAPTER_GGUF="${ADAPTER_GGUF:-$HOME/fttrain/adapter-all-lg-f16.gguf}"
LABEL="${LABEL:-gemma-4-e4b-all-lg-r16}"
REPO="${REPO:-$HOME/avtext/projects/avtext}"
SERVE="$HOME/scripts/avtext/serve.sh"
SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it/snapshots/*/ | head -1)
TB=llama-rocm-7.2.4_2

stop8080 () {
  p=$(ss -tlnHp 2>/dev/null | grep ':8080 ' | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "${p:-}" ] && { kill "$p" 2>/dev/null; for i in $(seq 1 20); do ss -tlnH | grep -q ':8080 ' || break; sleep 1; done; }
}
metrics () {  # $1 = model-id substring -> print the newest run's headline
  local d; d=$(ls -dt "$REPO"/reports/gemma-4/runs/*"$1"* 2>/dev/null | head -1)
  [ -n "$d" ] && python3 - "$d" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]+"/run.json"))
if "overall" in r:
    o=r["overall"]; print(f"  {r['model_id']}: value={o.get('recall',0):.1%} halluc={o.get('hallucination_rate',0):.1%} EM={o['exact_match']:.1%} invalid={r.get('n_invalid','?')}")
elif "metrics" in r:
    m=r["metrics"]; print(f"  {r['model_id']}: acc={m['accuracy']:.1%} macroF1={m['macro_f1']:.1%} invalid={m['invalid']}")
PY
}

echo "=== CHAIN_LG START $(date) | label=$LABEL ==="

# 0. convert grown adapter -> GGUF (idempotent)
if [ ! -f "$ADAPTER_GGUF" ]; then
  echo "=== convert adapter -> GGUF $(date +%H:%M) ==="
  toolbox run --container "$TB" bash -lc \
    "cd ~/gfx1151-fork && PYTHONPATH=~/gfx1151-fork/gguf-py ~/fttorch/bin/python convert_lora_to_gguf.py \
       '$ADAPTER_DIR' --base '$SNAP' --outfile '$ADAPTER_GGUF' --outtype f16" 2>&1 | tail -2
fi
[ -f "$ADAPTER_GGUF" ] || { echo "ADAPTER GGUF MISSING — abort"; exit 1; }

# 1. METAR — full eval/v2 (short outputs, serving stable)
echo "=== METAR eval/v2 $(date +%H:%M) ==="
stop8080; LORA="$ADAPTER_GGUF" "$SERVE" "$BASE" "$LABEL" || echo "serve FAILED"
( cd "$REPO" && uv run python -m avtext.harness.runner --completion --base-url http://127.0.0.1:8080 \
    --eval eval/v2/eval.jsonl --out-dir reports/gemma-4/runs \
    --model "$LABEL-metar" --model-id "$LABEL-metar" ) || echo "METAR EVAL FAILED"
stop8080; metrics "$LABEL-metar"

# 2. TAF — full eval/taf/v1
echo "=== TAF eval/taf/v1 $(date +%H:%M) ==="
stop8080; LORA="$ADAPTER_GGUF" "$SERVE" "$BASE" "$LABEL" || echo "serve FAILED"
( cd "$REPO" && uv run python -m avtext.harness.runner_taf --completion --base-url http://127.0.0.1:8080 \
    --eval eval/taf/v1/eval.jsonl --max-tokens 2048 --out-dir reports/gemma-4/runs \
    --model "$LABEL-taf" --model-id "$LABEL-taf" ) || echo "TAF EVAL FAILED"
stop8080; metrics "$LABEL-taf"

# 3. NOTAM — extraction (chunked, grammar-on by default) + classification (full)
echo "=== NOTAM ext(chunked,grammar)+cls $(date +%H:%M) ==="
BASE="$BASE" REPO="$REPO" bash "$HOME/scripts/avtext/eval_notam_chunked.sh" "$LABEL" "$ADAPTER_GGUF"
metrics "$LABEL-ext"; metrics "$LABEL-cls"

echo "=== CHAIN_LG DONE $(date) ==="
echo "collect with: cd $REPO && uv run python -m avtext.report.collect"