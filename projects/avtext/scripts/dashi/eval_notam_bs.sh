#!/usr/bin/env bash
# eval_notam_bs.sh — NOTAM base + single-task evals on the FULL sets (Session 22 / Phase 7 fix).
#
# The first NOTAM run used --limit 1500, which (because eval/notam/v1 is id-sorted and ids start
# with the category) covered only 4 of 11 categories. This re-evaluates the BASE model and the
# already-trained single-task adapter (gemma-4-e4b-notam-r16, saved from the killed run) on the
# FULL extraction (2,257, all categories) + FULL classification (4,047) sets, with the improved
# runner (raw-output capture, max_tokens=1024). The all-products adapter's NOTAM numbers come from
# phase_all.sh. Serial; same conventions as the other phase scripts.
#   base extraction is capped at 384 tokens (the untrained model rambles to the cap; this bounds its
#   ~0%-EM eval time) — the FT adapter emits short outputs that stop at EOS, so its cap is moot.
set -uo pipefail

BASE=~/models/gemma-4/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf
NOTAM_GGUF=~/ft/gemma-4/gemma-4-e4b-notam-r16-f16.gguf
REPO=~/avtext/projects/avtext
SERVE=~/scripts/avtext/serve.sh
EXT=eval/notam/v1/eval.jsonl
CLS=eval/notam_cls/v1/eval.jsonl

stop8080 () {
  p=$(ss -tlnHp 2>/dev/null | grep ':8080 ' | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "${p:-}" ] && { kill "$p" 2>/dev/null; for i in $(seq 1 20); do ss -tlnH | grep -q ':8080 ' || break; sleep 1; done; }
}
notam_ext () {  # $1=model id  $2=max-tokens
  ( cd "$REPO" && uv run python -m avtext.harness.runner_notam \
      --base-url http://127.0.0.1:8080 --eval "$EXT" --max-tokens "$2" \
      --out-dir reports/gemma-4/runs --model "$1" --model-id "$1" ) || echo "EXT eval FAILED for $1"
}
notam_cls () {  # $1=model id
  ( cd "$REPO" && uv run python -m avtext.harness.runner_notam \
      --base-url http://127.0.0.1:8080 --eval "$CLS" \
      --out-dir reports/gemma-4/runs --model "$1" --model-id "$1" ) || echo "CLS eval FAILED for $1"
}

echo "=== EVAL_NOTAM_BS START $(date) ==="
stop8080

# 1. Single-task FT adapter FIRST (the priority result — proper full-set numbers)
echo "=== SERVE notam-single $(date +%H:%M) ==="
LORA="$NOTAM_GGUF" "$SERVE" "$BASE" gemma-4-e4b-notam-r16 || echo "serve single FAILED"
echo "=== SINGLE EXT (FULL) $(date +%H:%M) ===";  notam_ext gemma-4-e4b-notam-r16-ext 1024
echo "=== SINGLE CLS (FULL) $(date +%H:%M) ===";  notam_cls gemma-4-e4b-notam-r16-cls
stop8080

# 2. BASE model (the before-picture; extraction capped to bound rambling)
echo "=== SERVE base $(date +%H:%M) ==="
"$SERVE" "$BASE" gemma-4-e4b-notam-base || echo "serve base FAILED"
echo "=== BASE EXT (FULL, cap 384) $(date +%H:%M) ===";  notam_ext gemma-4-e4b-notam-base-ext 384
echo "=== BASE CLS (FULL) $(date +%H:%M) ===";           notam_cls gemma-4-e4b-notam-base-cls
stop8080

# 3. RESULTS
echo "=== RESULTS $(date +%H:%M) ==="
cd "$REPO"
for id in notam-base-ext notam-r16-ext notam-base-cls notam-r16-cls; do
  d=$(ls -dt reports/gemma-4/runs/*"$id"* 2>/dev/null | grep -v 'smoke' | head -1)
  [ -n "$d" ] && python3 -c "
import json; r=json.load(open('$d/run.json'))
if 'overall' in r:
    o=r['overall']; print(f\"$id: value={o['recall'] or 0:.1%} halluc={o['hallucination_rate'] or 0:.1%} EM={o['exact_match']:.1%} n={o['n_records']} invalid={r.get('n_invalid')}\")
else:
    m=r['metrics']; print(f\"$id: accuracy={m['accuracy']:.1%} macro_f1={m['macro_f1']:.1%} n={m['n']} invalid={m['invalid']}\")"
done
echo "=== EVAL_NOTAM_BS DONE $(date) ==="
