#!/usr/bin/env bash
set -euo pipefail

# ====== Load config (single edit point) ======
source ./config.sh

# ====== Launch viewer ======
export VIEWER_HOST VIEWER_PORT VIEWER_SAVE_DIR="$MONITOR_DIR"
export VIEWER_TARGET_HEIGHT VIEWER_ADD_SEPARATOR VIEWER_SEPARATOR_WIDTH
python share/share_viewer.py > logs_viewer.txt 2>&1 &
VIEWER_PID=$!
echo "[run_all] viewer pid=$VIEWER_PID"

cleanup() {
  if kill -0 "$VIEWER_PID" >/dev/null 2>&1; then
    kill "$VIEWER_PID" || true
  fi
}
trap cleanup EXIT

# ====== Launch pipeline ======
COMMON_ARGS=(
  --mode "$MODE"
  --monitor_dir "$MONITOR_DIR"
  --supervisor_backend "$SUPERVISOR_BACKEND"
  --supervisor_model_path "$SUPERVISOR_MODEL_PATH"
  --supervisor_lora_path "$SUPERVISOR_LORA_PATH"
  --supervisor_loop_interval_sec "$SUPERVISOR_LOOP_INTERVAL_SEC"
  --robot_url "$ROBOT_URL"
  --out_action_file "$OUT_ACTION_FILE"
  --log_json "$LOG_JSON"
)

INTERACTIVE_FLAG=()
if [[ "${INTERACTIVE_SELECT}" == "1" ]]; then
  INTERACTIVE_FLAG=(--interactive_select)
fi

if [[ "$MODE" == "sequential" ]]; then
  python bigbrain/pipeline_runner.py \
    "${COMMON_ARGS[@]}" \
    --ckpt_or_lora_dir "$CKPT_OR_LORA_DIR" \
    --base_model "$BASE_MODEL"
elif [[ "$MODE" == "3dw" ]]; then
  python bigbrain/pipeline_runner.py \
    "${COMMON_ARGS[@]}" \
    --mode 3dw \
    --ckpt_or_lora_dir "$CKPT_OR_LORA_DIR" \
    --base_model "$BASE_MODEL" \
    --user_instruction "$USER_INSTRUCTION" \
    --pairwise_prompt_jsonl "$PAIRWISE_PROMPT_JSONL" \
    --pairwise_prompt_text "$PAIRWISE_PROMPT_TEXT" \
    "${INTERACTIVE_FLAG[@]}"
else
  python bigbrain/pipeline_runner.py \
    "${COMMON_ARGS[@]}" \
    --user_instruction "$USER_INSTRUCTION" \
    "${INTERACTIVE_FLAG[@]}"
fi
