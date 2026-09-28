#!/bin/bash
# ADR-001 Gate 2 step 4 ("halve the adapter scale and re-evaluate"), queued behind the main window
# and the MTP window: waits until the MTP window has restored OCR (its last log line) and exited,
# then runs every public benchmark once more at LoRA scale 0.5 (state adapter_half).
RUN=/home/wdenejko/benchlab/runs/2026-09-24-gate2-battery
until grep -qs "OCR ->" ~/benchlab/logs/battery-mtp-2*.log && ! pgrep -f "[b]attery_mtp.sh" >/dev/null \
      && ! pgrep -f "[b]attery_window.sh" >/dev/null; do sleep 60; done
PLAN="humaneval_plus:adapter_half ifeval:adapter_half ds1000:adapter_half bfcl:adapter_half bird:adapter_half mmlu_pro:adapter_half lcb:adapter_half"
exec ~/benchlab/scripts/battery/battery_window.sh $RUN "$PLAN"
