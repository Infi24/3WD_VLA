#!/usr/bin/env bash
set -euo pipefail

# ====== Load config (single edit point) ======
source ./config.sh

VIEWER_PID=""
BRIDGE_PID=""

cleanup() {
  if [[ -n "${VIEWER_PID}" ]] && kill -0 "$VIEWER_PID" >/dev/null 2>&1; then
    kill "$VIEWER_PID" || true
  fi
  if [[ "${KEEP_BRIDGE_ON_EXIT:-1}" != "1" ]] && [[ -n "${BRIDGE_PID}" ]] && kill -0 "$BRIDGE_PID" >/dev/null 2>&1; then
    kill "$BRIDGE_PID" || true
  fi
}
trap cleanup EXIT

INTERACTIVE_FLAG=()
if [[ "${INTERACTIVE_SELECT}" == "1" ]]; then
  INTERACTIVE_FLAG=(--interactive_select)
fi

INITIAL_IMAGE_FLAG=()
if [[ "${PREFER_NAMED_INITIAL_IMAGE}" == "1" ]]; then
  INITIAL_IMAGE_FLAG=(--prefer_named_initial_image --initial_image_name "$INITIAL_IMAGE_NAME")
fi

SUPERVISOR_FOLLOWUP_FLAG=()
if [[ "${SUPERVISOR_ONLY_AFTER_FIRST}" == "1" ]]; then
  SUPERVISOR_FOLLOWUP_FLAG=(
    --supervisor_only_after_first
    --supervisor_only_timeout_sec "$SUPERVISOR_ONLY_TIMEOUT_SEC"
    --supervisor_only_max_images "$SUPERVISOR_ONLY_MAX_IMAGES"
  )
fi

DISABLE_SUPERVISOR_LOOP_FLAG=()
if [[ "${DISABLE_SUPERVISOR_LOOP:-0}" == "1" ]]; then
  DISABLE_SUPERVISOR_LOOP_FLAG=(--disable_supervisor_loop)
fi

SKIP_INITIAL_SUPERVISOR_FLAG=()
if [[ "${SKIP_INITIAL_SUPERVISOR:-0}" == "1" ]]; then
  SKIP_INITIAL_SUPERVISOR_FLAG=(--skip_initial_supervisor)
fi

if [[ "${START_VIEWER_IN_RUN_3DW}" == "1" ]]; then
  export VIEWER_HOST VIEWER_PORT VIEWER_SAVE_DIR="$MONITOR_DIR"
  export VIEWER_TARGET_HEIGHT VIEWER_ADD_SEPARATOR VIEWER_SEPARATOR_WIDTH
  export VIEWER_CLEAR_ON_START="${CLEAR_SHARE_PHOTO_ON_START:-1}"
  python share/share_viewer.py > logs_viewer_3dw.txt 2>&1 &
  VIEWER_PID=$!
  echo "[run_3dw] viewer pid=$VIEWER_PID"
fi

if [[ "${START_BRIDGE_IN_RUN_3DW}" == "1" ]]; then
  BRIDGE_CLEAR_FLAG=()
  if [[ "${CLEAR_SHARE_PHOTO_ON_START:-0}" == "1" ]]; then
    BRIDGE_CLEAR_FLAG=(--clear_monitor_dir)
  fi
  python -u -m share.bridge_http \
    --host "$BRIDGE_HOST" \
    --port "$BRIDGE_PORT" \
    --prompt_file "$BRIDGE_PROMPT_FILE" \
    --monitor_dir "$BRIDGE_MONITOR_DIR" \
    "${BRIDGE_CLEAR_FLAG[@]}" \
    > logs_bridge_3dw.txt 2>&1 &
  BRIDGE_PID=$!
  echo "[run_3dw] bridge pid=$BRIDGE_PID"
fi

python bigbrain/pipeline_runner.py \
  --mode 3dw \
  --monitor_dir "$MONITOR_DIR" \
  --wait_image_timeout_sec "$WAIT_IMAGE_TIMEOUT_SEC" \
  --ckpt_or_lora_dir "$CKPT_OR_LORA_DIR" \
  --base_model "$BASE_MODEL" \
  --supervisor_backend "$SUPERVISOR_BACKEND" \
  --supervisor_model_path "$SUPERVISOR_MODEL_PATH" \
  --supervisor_lora_path "$SUPERVISOR_LORA_PATH" \
  --supervisor_loop_interval_sec "$SUPERVISOR_LOOP_INTERVAL_SEC" \
  --robot_url "$ROBOT_URL" \
  --out_action_file "$OUT_ACTION_FILE" \
  --out_robot_prompt_file "$OUT_ROBOT_PROMPT_FILE" \
  --log_json "$LOG_JSON" \
  --user_instruction "$USER_INSTRUCTION" \
  --pairwise_prompt_jsonl "$PAIRWISE_PROMPT_JSONL" \
  --pairwise_prompt_text "$PAIRWISE_PROMPT_TEXT" \
  "${INTERACTIVE_FLAG[@]}" \
  "${INITIAL_IMAGE_FLAG[@]}" \
  "${SUPERVISOR_FOLLOWUP_FLAG[@]}" \
  "${DISABLE_SUPERVISOR_LOOP_FLAG[@]}" \
  "${SKIP_INITIAL_SUPERVISOR_FLAG[@]}"
