#!/bin/bash
# A Revision 2 checkpoint, made ready for the mini-battery (ADR-004 Revision 2, action item 9).
# Needs no GPU:
#   checkpoint.sh TRAIN_RUN STEP CAL_RUN CKPT_RUN      (STEP: a checkpoint's number, or final)
# - the checkpoint's adapter, exported to a llama.cpp LoRA GGUF (dsbench.sftgen.export_lora_gguf,
#   in the training venv): CKPT_RUN/lora.gguf;
# - CKPT_RUN, whose base passes are the calibration's: its items, data, gold scores and the base
#   and A/A passes are links into CAL_RUN, read and never written. Only the checkpoint's own
#   passes are new.
# Then a battery window answers the checkpoint's passes (state `adapter`, LoRA scale 1):
#   ARM=1 CTX=196608 MAX_HOURS=9 LORA=CKPT_RUN/lora.gguf nohup setsid \
#     ~/benchlab/scripts/battery/battery_window.sh CKPT_RUN \
#     "bfcl:adapter ifeval:adapter humaneval_plus:adapter bird:adapter" </dev/null >/dev/null 2>&1 &
# and from ~/benchlab/scripts/battery/src, with the battery's venv:
#   python -m dsbench.battery.score --run-dir CKPT_RUN --bench ifeval,bfcl,bird,humaneval_plus \
#     --states adapter
#   python -m dsbench.battery.mini summary --run-dir CKPT_RUN --state adapter --aa base_rep ...
set -eu
TRAIN=$1; STEP=$2; CAL=$3; OUT=$4
SRC=~/benchlab/scripts/battery/src  # a synced copy of the repo's src/ (dsbench/)
if [ "$STEP" = final ]; then CKPT=$TRAIN/out_qwen36_35b_rev2/final
else CKPT=$TRAIN/out_qwen36_35b_rev2/checkpoint-$STEP; fi
[ -f "$CKPT/adapter_model.safetensors" ] || { echo "no adapter in $CKPT" >&2; exit 1; }
[ -e "$OUT" ] && { echo "$OUT exists: not touching it" >&2; exit 1; }
mkdir -p "$OUT/gen" "$OUT/scores"
ln -s "$(readlink -f "$CAL/items")" "$OUT/items"
ln -s "$(readlink -f "$CAL/data")" "$OUT/data"
for f in "$CAL"/gen/*.base.jsonl "$CAL"/gen/*.base_rep.jsonl; do
  [ -e "$f" ] && ln -s "$(readlink -f "$f")" "$OUT/gen/"
done
for f in "$CAL"/scores/*.gold.jsonl "$CAL"/scores/*.base.jsonl "$CAL"/scores/*.base_rep.jsonl; do
  [ -e "$f" ] && ln -s "$(readlink -f "$f")" "$OUT/scores/"
done
echo "$(readlink -f "$CKPT")" >"$OUT/CHECKPOINT"
toolbox run -c llama-rocm-unlimited-build bash -lc "source ~/ftgguf/bin/activate
  cd $SRC && PYTHONPATH=. python -m dsbench.sftgen.export_lora_gguf --adapter $CKPT \
    --out $OUT/lora.gguf" </dev/null
ls -la "$OUT" "$OUT/gen" "$OUT/scores"
