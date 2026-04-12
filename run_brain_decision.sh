#!/usr/bin/env bash
set -euo pipefail

# ====== Load config (single edit point) ======
source ./config.sh

# ====== Run one-step brain decision (single image) ======
# This script is for the classic sequential style: call once per step.
# It uses the latest merged image from MONITOR_DIR.
LATEST_IMG="$(ls -t "$MONITOR_DIR"/*.jpg "$MONITOR_DIR"/*.png "$MONITOR_DIR"/*.jpeg 2>/dev/null | head -n 1)"
if [[ -z "$LATEST_IMG" ]]; then
  echo "[ERROR] No image found in $MONITOR_DIR"
  exit 1
fi

python bigbrain/brain_decision.py \
  --ckpt_or_lora_dir "$CKPT_OR_LORA_DIR" \
  --base_model "$BASE_MODEL" \
  --instruction_lib "$INSTRUCTION_LIB" \
  --image_path "$LATEST_IMG" \
  --out_action_file "$OUT_ACTION_FILE"
