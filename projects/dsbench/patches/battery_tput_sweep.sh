#!/bin/bash
# Throughput sweep for the battery server: the same 128 IFEval prompts (max_tokens 512, adapter
# state) at several slot counts. MoE decode on this APU is weight-read bound, and 8 concurrent
# tokens already touch ~half the experts, so more slots should add throughput almost for free.
# Launch detached; OCR is stopped for the sweep and restarted on exit.
set -u
RUN=$1; shift; SWEEP=${*:-"8 16 32"}
LOG=~/benchlab/logs/battery-tput-$(date +%Y%m%d-%H%M%S).log
PY=~/benchlab/batteryvenv/bin/python; SRC=~/benchlab/scripts/battery/src; URL=http://127.0.0.1:8093
SRV_PAT='^/home/wdenejko/src/llama-qwen4exp-src/build-v2/bin/llama-server .*--port 8093'
stop_server(){ pkill -f "$SRV_PAT"; for _ in $(seq 1 30); do pgrep -f "$SRV_PAT" >/dev/null || return 0; sleep 2; done; pkill -9 -f "$SRV_PAT"; }
restore(){ stop_server; systemctl --user start dashi-unlimited-ocr.service; echo "OCR -> $(systemctl --user is-active dashi-unlimited-ocr.service)" >>"$LOG"; }
trap restore EXIT
systemctl --user stop dashi-unlimited-ocr.service; sleep 3
source ~/fttrain/gttwait.sh
mkdir -p $RUN/tput/items
$PY - "$RUN" <<'PYEOF'
import json, sys
run = sys.argv[1]
rows = [json.loads(l) for l in open(f"{run}/items/ifeval.jsonl")][:128]
with open(f"{run}/tput/items/probe.jsonl", "w") as fh:
    for r in rows:
        r["gen"]["max_tokens"] = 512
        fh.write(json.dumps(r) + "\n")
PYEOF
for np in $SWEEP; do
  stop_server; gttwait >>"$LOG" 2>&1
  NP=$np nohup ~/benchlab/scripts/battery/battery_server.sh >>~/benchlab/logs/battery-tput-server.log 2>&1 </dev/null &
  for _ in $(seq 1 120); do curl -sf $URL/health >/dev/null && break; sleep 5; done
  t0=$(date +%s)
  (cd $SRC && PYTHONPATH=. $PY -m dsbench.battery.generate --items $RUN/tput/items/probe.jsonl \
     --state adapter --out $RUN/tput/np$np.jsonl --workers $np) >>"$LOG" 2>&1
  t1=$(date +%s)
  toks=$($PY -c "import json,sys; print(sum(json.loads(l).get('completion_tokens') or 0 for l in open('$RUN/tput/np$np.jsonl')))")
  busy=$(curl -s $URL/metrics | awk '/^llamacpp:n_busy_slots_per_decode/{print $2}')
  echo "RESULT np=$np tokens=$toks wall_s=$((t1-t0)) tok_per_s=$((toks/(t1-t0))) busy_per_decode=$busy" >>"$LOG"
done
