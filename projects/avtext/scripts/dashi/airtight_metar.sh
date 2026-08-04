#!/usr/bin/env bash
# airtight_metar.sh — re-run the SINGLE-TASK METAR adapter (v2-r16) on the SAME 1,500 records the
# combined adapter used, so single-vs-combined on METAR is a strict same-N comparison (matching the
# already-airtight TAF comparison). ~1.3h on this iGPU.
set -uo pipefail
BASE=~/models/gemma-4/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf
LORA_GGUF=~/ft/gemma-4/gemma-4-e4b-v2-r16-f16.gguf
REPO=~/avtext/projects/avtext
SERVE=~/scripts/avtext/serve.sh
LIMIT=1500
stop8080 () { p=$(ss -tlnHp 2>/dev/null | grep ':8080 ' | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "${p:-}" ] && { kill "$p" 2>/dev/null; sleep 5; }; }

echo "=== AIRTIGHT METAR START $(date) ==="
stop8080
LORA="$LORA_GGUF" "$SERVE" "$BASE" gemma-4-e4b-v2-r16 || echo "serve FAILED"
echo "=== EVAL single-task METAR on eval/v2 (limit $LIMIT) $(date +%H:%M) ==="
( cd "$REPO" && uv run python -m avtext.harness.runner --completion --base-url http://127.0.0.1:8080 \
    --eval eval/v2/eval.jsonl --limit "$LIMIT" --out-dir reports/gemma-4/runs \
    --model gemma-4-e4b-v2-r16-metar1500 --model-id gemma-4-e4b-v2-r16-metar1500 ) || echo "EVAL FAILED"
stop8080
echo "=== RESULT ==="
cd "$REPO"; D=$(ls -dt reports/gemma-4/runs/*v2-r16-metar1500* 2>/dev/null | head -1)
[ -n "$D" ] && python3 -c "import json; r=json.load(open('$D/run.json')); o=r['overall']; print(f'single-METAR(1500): value={o[\"recall\"]:.1%} halluc={o[\"hallucination_rate\"]:.1%} EM={o[\"exact_match\"]:.1%} invalid={r[\"n_invalid\"]}')"
echo "=== AIRTIGHT METAR DONE $(date) ==="
