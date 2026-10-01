#!/bin/bash
# GPU window for Target A's reasoning-prefill pilot (ADR-004 Revision 2, action item 2). Launch
# detached, after the owner has stopped production:
#   nohup setsid ~/benchlab/scripts/target-a-prefill-pilot/window.sh </dev/null >/dev/null 2>&1 &
# The bare base (I-Mini, no LoRA) answers the ClickHouse weekday and weekend rows in thinking mode,
# in two phases, through dsbench.sftgen.reasoning_pilot generate (which resumes where a previous
# run stopped):
#   1. items.jsonl (target_a_hints prefill-items): each row 4 times plain, and 4 times with its
#      thinking block opened by the convention (`start`);
#   2. items_recall.jsonl (dsbench.sftgen.prefill splice, run here between the phases): each plain
#      reply cut where it first turns to the weekday function, the convention written there, and
#      the base continuing from it (`recall`).
# Every item goes through the raw completion path (/apply-template, then /completion), the plain
# ones with nothing prefilled. The Mac then checks the replies (target_a_hints verify). As
# target_a_hints_window.sh does, this refuses to start while production (any llama-server but
# OCR's) runs, and handles OCR as ~/benchlab/RULES.md says: stopped once it has finished loading,
# and restarted on exit only if it was running when the window began.
set -u
D=~/benchlab/scripts/target-a-prefill-pilot
STAMP=$(date +%Y%m%d-%H%M%S)
L=~/benchlab/logs; LOG=$L/target-a-prefill-window-$STAMP.log
URL=http://127.0.0.1:8093
SRV_PAT='^/home/wdenejko/src/llama-qwen4exp-src/build-v2/bin/llama-server .*--port 8093'
OCR_UNIT=dashi-unlimited-ocr.service
OCR_PAT='^/home/wdenejko/src/llama-unlimited-ocr/[^ ]*/llama-server '
PY=~/benchlab/batteryvenv/bin/python
mkdir -p "$L"
log(){ echo "[window] $(date +%H:%M:%S) $*" >>"$LOG"; }
production(){ pgrep -af '^[^ ]*/llama-server( |$)' | grep -qv 'llama-unlimited-ocr/'; }
if production; then log "a llama-server other than OCR's is running: not starting"; exit 1; fi
# Stopping OCR while it is still loading leaves its llama-server stuck inside the container,
# holding GPU memory (2026-09-25): wait for its /health first, and kill a server that outlives the
# stop.
stop_ocr(){
  if [ "$(systemctl --user is-active $OCR_UNIT)" = active ]; then
    for _ in $(seq 1 24); do curl -sf http://127.0.0.1:8144/health >/dev/null && break; sleep 5; done
  fi
  systemctl --user stop $OCR_UNIT; sleep 3
  if pgrep -f "$OCR_PAT" >/dev/null; then
    pkill -f "$OCR_PAT"; sleep 10; pkill -9 -f "$OCR_PAT"; log "killed an OCR server that outlived the stop"
  fi; }
stop_server(){ pkill -f "$SRV_PAT"; for _ in $(seq 1 30); do pgrep -f "$SRV_PAT" >/dev/null || return 0; sleep 2; done; pkill -9 -f "$SRV_PAT"; }
OCR_WAS=$(systemctl --user is-active $OCR_UNIT)
restore(){ stop_server; [ -n "${GOV:-}" ] && kill "$GOV" 2>/dev/null
  if [ "$OCR_WAS" = active ]; then systemctl --user start $OCR_UNIT; sleep 2; fi
  log "OCR -> $(systemctl --user is-active $OCR_UNIT) (was $OCR_WAS)"; log "window exit"; }
trap restore EXIT
log "stopping OCR (was $OCR_WAS)"
stop_ocr
source ~/fttrain/gttwait.sh; gttwait >>"$LOG" 2>&1
# 8 slots share one KV pool: 8 x (prompt + 16k-token reply) needs ~150k tokens, so 196k.
NOLORA=1 CTX=196608 NP=8 nohup $D/battery_server.sh >>$L/target-a-prefill-server-$STAMP.log 2>&1 </dev/null &
for _ in $(seq 1 120); do curl -sf $URL/health >/dev/null && break; sleep 5; done
curl -sf $URL/health >/dev/null || { log "server never became healthy"; exit 1; }
log "server up (base, no LoRA)"
# About an hour of generation: under the thermal governor, anchored on this server.
PATTERN="$SRV_PAT" nohup ~/fttrain/thermostat.sh >>$L/target-a-prefill-thermostat-$STAMP.log 2>&1 </dev/null &
GOV=$!
generate(){  # items file -> replies file, both in $D
  (cd $D && PYTHONPATH=$D/src $PY -m dsbench.sftgen.reasoning_pilot generate --items $D/$1 \
     --out $D/$2 --base-url $URL --workers 8 --max-tokens 16384) \
     >>$L/target-a-prefill-generate-$STAMP.log 2>&1
}
records(){ echo "$(wc -l < $D/$1) records, $(grep -c '"error": ""' $D/$1) without error"; }
# Smoke test of the prefilled path: the template must open a thinking block, the base must
# continue the prefill and close it, and </think> must come back in the text. A reply that keeps
# its prefill at the head of the reasoning, finishes its turn and has an answer shows all three.
echo '{"id": "smoke:prefill", "pool": "smoke", "messages": [{"role": "user", "content": "Is 91 prime? Answer yes or no."}], "prefill": "91 = 7 x 13."}' >$D/smoke_items.jsonl
generate smoke_items.jsonl smoke_gen.jsonl
if ! $PY -c 'import json,sys; r=json.loads(open(sys.argv[1]).readlines()[-1]); sys.exit(0 if not r["error"] and r["finish_reason"] == "stop" and r["reasoning"].startswith("91 = 7 x 13.") and r["answer"] else 1)' $D/smoke_gen.jsonl; then
  log "smoke test: the prefilled reply didn't come back whole; stopping"; exit 1; fi
log "smoke test ok: the prefill was continued and the thinking closed"
log "phase 1 start: items.jsonl (plain and start) -> gen.jsonl"
generate items.jsonl gen.jsonl
log "phase 1 end: $(records gen.jsonl)"
(cd $D && PYTHONPATH=$D/src $PY -m dsbench.sftgen.prefill splice --items $D/items.jsonl \
   --gen $D/gen.jsonl --out $D/items_recall.jsonl) >>"$LOG" 2>&1
log "spliced: $(wc -l < $D/items_recall.jsonl) recall items"
log "phase 2 start: items_recall.jsonl -> gen_recall.jsonl"
generate items_recall.jsonl gen_recall.jsonl
log "phase 2 end: $(records gen_recall.jsonl)"
log "window done"
