#!/bin/bash
# Gate-2 LoRA training window (ADR-001 Gate 2). Detached-safe: launch with nohup setsid so it
# outlives the ssh session. QWEN35_RESUME=1 in the environment resumes from the last checkpoint.
set -u
G=~/gate2; LOG=$G/train.log
PAT='^([^ ]*/)?python[0-9.]* +train_qwen3_5_35b\.py'
restore(){
  [ -n "${THERMO:-}" ] && kill "$THERMO" 2>/dev/null
  systemctl --user start dashi-unlimited-ocr.service 2>>"$LOG"; sleep 2
  echo "[window] $(date +%H:%M:%S) OCR -> $(systemctl --user is-active dashi-unlimited-ocr.service)" >>"$LOG"
}
trap restore EXIT
echo "[window] $(date +%H:%M:%S) stopping OCR for the training window" >>"$LOG"
systemctl --user stop dashi-unlimited-ocr.service 2>>"$LOG"; sleep 3
source ~/fttrain/gttwait.sh; gttwait                 # a just-exited GPU process can hold 80-95 GiB
echo "[window] $(date +%H:%M:%S) gtt drained; starting thermostat + trainer" >>"$LOG"
PATTERN="$PAT" nohup ~/fttrain/thermostat.sh >>$G/thermostat.log 2>&1 &
THERMO=$!
toolbox run -c llama-rocm-unlimited-build bash -lc "
  source ~/ftgguf/bin/activate
  export PYTHONPATH=\$HOME/src/aiter PYTORCH_ROCM_ARCH=gfx1151 GPU_ARCHS=gfx1151 \
         QWEN35_ATTN_IMPL=kernels-community/aiter-flash-attn \
         QWEN35_REPORT_TO=none QWEN35_MAX_STEPS=-1 WANDB_DISABLED=true \
         QWEN35_DATASET_DIR=$G/data_tokenized_gate2 QWEN35_OUTPUT_DIR=$G/out_qwen36_35b_gate2 \
         QWEN35_RESUME=${QWEN35_RESUME:-0} \
         TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 HIP_FORCE_DEV_KERNARG=1 \
         PYTORCH_ALLOC_CONF=expandable_segments:True
  cd ~/src/transformers5-qwen3.5-recipe
  python train_qwen3_5_35b.py
" </dev/null >>"$LOG" 2>&1
echo "[window] $(date +%H:%M:%S) trainer exited rc=$?" >>"$LOG"
