#!/bin/bash
# GPU window for the seq-4096 enablement (ADR-001 retrain, 2026-09-28). Launch detached:
#   nohup setsid ~/benchlab/scripts/seq4096_window.sh </dev/null >/dev/null 2>&1 &
# 1. validate: the 38 new MMQ keys (and their seq-2048 twins) against torch-ggml-ops' oracle;
# 2. the recipe's training-step audit (rank 4, 2 steps, via audit_long_seq.py) at seq 2048 (the
#    control: Gate 0 measured 6.91 s/step and 15.26 GiB reserved), 4096 (the new MMQ keys and GMM
#    entries; only if step 1 passed) and 8192 (batch-1 seq-8192 has batch-4 seq-2048's shapes,
#    which the existing bundle and GMM tables already cover).
# The owner stops production before a GPU window; this refuses to start while any llama-server
# or the OCR unit is up, rather than stopping them itself.
set -u
STAMP=$(date +%Y%m%d-%H%M%S)
L=~/benchlab/logs; LOG=$L/seq4096-window-$STAMP.log
log(){ echo "[window] $(date +%H:%M:%S) $*" >>"$LOG"; }
# anchored on the executable: a loose "llama-server" also matches any shell whose command line
# merely mentions it (the launching ssh session did, on the first try)
if pgrep -f '^[^ ]*/llama-server( |$)' >/dev/null; then
  log "a llama-server is running: not starting"; exit 1; fi
if [ "$(systemctl --user is-active dashi-unlimited-ocr.service)" = active ]; then
  log "OCR is running: not starting"; exit 1; fi
source ~/fttrain/gttwait.sh; gttwait >>"$LOG" 2>&1
PATTERN='^([^ ]*/)?python[0-9.]* +[^ ]*(audit_long_seq|validate_seq4096_keys)\.py' \
  nohup ~/fttrain/thermostat.sh >>$L/seq4096-thermostat-$STAMP.log 2>&1 </dev/null &
GOV=$!; trap 'kill $GOV 2>/dev/null; log "window exit"' EXIT
stage(){  # stage NAME COMMAND: one GPU process in the training toolbox, venv active
  log "start $1"
  toolbox run -c llama-rocm-unlimited-build bash -lc "
    source ~/ftgguf/bin/activate
    export PYTHONPATH=\$HOME/src/transformers5-qwen3.5-recipe:\$HOME/src/aiter \
           PYTORCH_ROCM_ARCH=gfx1151 GPU_ARCHS=gfx1151 QWEN35_ATTN_IMPL=kernels-community/aiter-flash-attn \
           TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 HIP_FORCE_DEV_KERNARG=1 \
           PYTORCH_ALLOC_CONF=expandable_segments:True
    $2" >>$L/seq4096-$1-$STAMP.log 2>&1
  local rc=$?; log "end $1 (exit $rc)"; gttwait >>"$LOG" 2>&1; return $rc
}
audit(){ stage audit$1 "cd ~/src/transformers5-qwen3.5-recipe && python ~/benchlab/scripts/audit_long_seq.py \
  --dataset-dir ~/src/transformers5-qwen3.5-recipe/data_tokenized_qwen3.5 --rank 4 --alpha 4 \
  --sequence-length $1 --max-steps 2 --report-output $L/seq4096-audit$1-$STAMP.json"; }
stage validate "cd ~/src/torch-ggml-ops && python ~/benchlab/scripts/validate_seq4096_keys.py $L/seq4096-validate-$STAMP.json"
# Gate on the summary, not the exit status: a Python process that initialized HIP exits 0 on this
# box whatever sys.exit says (found 2026-09-28, when this gate let a failed validation through).
VALID=$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1]))["summary"]["new"]; print("yes" if s["pass"] and not s["FAIL"] else "no")' \
  $L/seq4096-validate-$STAMP.json 2>/dev/null || echo no)
audit 2048
if [ "$VALID" = yes ]; then audit 4096; else log "skip audit4096: the new keys failed validation"; fi
audit 8192
log "window done"
