#!/usr/bin/env bash
set -euo pipefail

# ====== Load config (single edit point) ======
source ./config.sh

# ====== Run selector-only pipeline ======
python bigbrain/pipeline_runner.py \
  --mode selector \
  --monitor_dir "$MONITOR_DIR" \
  --supervisor_backend "$SUPERVISOR_BACKEND" \
  --supervisor_model_path "$SUPERVISOR_MODEL_PATH" \
  --supervisor_lora_path "$SUPERVISOR_LORA_PATH" \
  --supervisor_loop_interval_sec "$SUPERVISOR_LOOP_INTERVAL_SEC" \
  --robot_url "$ROBOT_URL" \
  --out_action_file "$OUT_ACTION_FILE" \
  --log_json "$LOG_JSON" \
  --user_instruction "$USER_INSTRUCTION" \
  $( [ "${INTERACTIVE_SELECT}" = "1" ] && echo "--interactive_select" )
