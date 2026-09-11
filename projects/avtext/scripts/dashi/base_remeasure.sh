#!/usr/bin/env bash
# base_remeasure.sh — re-score the SERVED BASE (no LoRA) under exactly the protocol used for the S24
# grown adapter, so the model card's before/after table is one protocol per row:
#   1. TAF   full eval/taf/v1 (5,294), raw /completion, greedy, 2048-token cap   (base had only a 1,500 smoke at 1024)
#   2. NOTAM extraction, 20 records per server restart (base run was 100/restart, the degraded protocol)
#   3. METAR eval/v2 (6,200), raw /completion, 256-token cap                     (base run predates the completion path)
# Run ON THE HOST (wraps toolbox via serve.sh). Resumable: runners checkpoint predictions.
set -uo pipefail
BASE="${BASE:-$HOME/fttrain/gemma-4-E4B-it-Q8_0.gguf}"; REPO="${REPO:-$HOME/avtext/projects/avtext}"; SERVE="$HOME/scripts/avtext/serve.sh"
stop8080 () { p=$(ss -tlnHp 2>/dev/null | grep ':8080 ' | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "${p:-}" ] && { kill "$p" 2>/dev/null; for i in $(seq 1 20); do ss -tlnH | grep -q ':8080 ' || break; sleep 1; done; }; }
headline () {  # $1 = run-dir glob
  local d; d=$(ls -dt $1 2>/dev/null | grep -v smoke | head -1); [ -n "$d" ] && python3 - "$d" <<'PY'
import json,sys; r=json.load(open(sys.argv[1]+"/run.json"))
if "overall" in r:
    o=r["overall"]; print(f"BASERESULT {r['model_id']} n={o['n_records']} decode={r.get('decode_params')}: value={o.get('recall',0):.1%} halluc={o.get('hallucination_rate',0):.1%} EM={o['exact_match']:.1%} invalid={r.get('n_invalid','?')}")
else:
    m=r["metrics"]; print(f"BASERESULT {r['model_id']}: acc={m['accuracy']:.1%} macroF1={m['macro_f1']:.1%} invalid={m['invalid']}")
PY
}
source $HOME/fttrain/gttwait.sh
echo "=== BASE REMEASURE START $(date) ==="
# 1. TAF full @2048
gttwait; echo "=== TAF base full @2048 $(date +%H:%M) ==="
stop8080; "$SERVE" "$BASE" gemma-4-e4b-taf-base || echo "serve FAILED"
( cd "$REPO" && uv run python -m avtext.harness.runner_taf --completion --base-url http://127.0.0.1:8080 \
    --eval eval/taf/v1/eval.jsonl --max-tokens 2048 --out-dir reports/gemma-4/runs \
    --model gemma-4-e4b-taf-base --model-id gemma-4-e4b-taf-base ) || echo "TAF BASE FAILED"
stop8080; headline "$REPO/reports/gemma-4/runs/*gemma-4-e4b-taf-base"
# 2. NOTAM extraction, base, 20/restart (ext only) — reuse notam_ext_c20.sh with an empty LoRA
gttwait; echo "=== NOTAM ext base c20 $(date +%H:%M) ==="
LABEL=gemma-4-e4b-notam-base; CHUNKDIR=reports/gemma-4/runs/chunks/${LABEL}-c20-ext; cd "$REPO"; rm -rf "$CHUNKDIR"
off=0; ci=0
while [ "$off" -lt 2257 ]; do
  stop8080; "$SERVE" "$BASE" "$LABEL" >/dev/null || echo "serve fail chunk $ci"
  uv run python -m avtext.harness.runner_notam --base-url http://127.0.0.1:8080 \
    --eval eval/notam/v1/eval.jsonl --offset "$off" --limit 20 --max-tokens 512 \
    --out-dir "$CHUNKDIR" --out-name "$(printf 'c%03d' "$ci")" \
    --model "$LABEL-c20-ext" --model-id "$LABEL-c20-ext" >/dev/null 2>&1 || echo "chunk $ci FAILED"
  [ $((ci % 20)) -eq 0 ] && echo "--- base ext-c20 chunk $ci/113 $(date +%H:%M) ---"
  off=$((off + 20)); ci=$((ci + 1))
done
stop8080
uv run python -m avtext.report.merge_notam --chunks "$CHUNKDIR" --out-dir reports/gemma-4/runs --model-id "$LABEL-c20-ext" || echo "MERGE FAILED"
headline "$REPO/reports/gemma-4/runs/*$LABEL-c20-ext*"
# 3. METAR v2 full, completion endpoint, 256 cap
gttwait; echo "=== METAR base v2 completion $(date +%H:%M) ==="
stop8080; "$SERVE" "$BASE" gemma-4-e4b-metar-base || echo "serve FAILED"
( cd "$REPO" && uv run python -m avtext.harness.runner --completion --base-url http://127.0.0.1:8080 \
    --eval eval/v2/eval.jsonl --out-dir reports/gemma-4/runs --max-tokens 256 \
    --model gemma-4-e4b-metar-base --model-id gemma-4-e4b-metar-base ) || echo "METAR BASE FAILED"
stop8080; headline "$REPO/reports/gemma-4/runs/*gemma-4-e4b-metar-base"
echo "=== BASE REMEASURE DONE $(date) ==="
