#!/usr/bin/env bash
set -euo pipefail

# Single place to edit all paths and settings for deployment.

# Mode: sequential | selector | 3dw
MODE="3dw"

# Paths
MONITOR_DIR="/root/autodl-tmp/share_photo"
OFFLINE_MONITOR_DIR="/root/autodl-tmp/test_images_merged"
CKPT_OR_LORA_DIR="/root/autodl-tmp/bigbrain/saves/qwen3_8b_grasp_final/checkpoint-2700"
BASE_MODEL="/root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct"
INSTRUCTION_LIB="/root/autodl-tmp/share/instruction_library.py"
OUT_ACTION_FILE="/root/autodl-tmp/share_cmd/action_code.txt"
OUT_ROBOT_PROMPT_FILE="/root/autodl-tmp/share_cmd/robot_prompt.txt"
LOG_JSON="/root/autodl-tmp/logs/pipeline_result.json"
WAIT_IMAGE_TIMEOUT_SEC="600"

# Supervisor backend: api | local
SUPERVISOR_BACKEND="local"
SUPERVISOR_MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct"
SUPERVISOR_LORA_PATH=""
SUPERVISOR_LOOP_INTERVAL_SEC="5"

# Viewer (image receiver)
VIEWER_HOST="0.0.0.0"
VIEWER_PORT="9999"
VIEWER_TARGET_HEIGHT="384"
VIEWER_ADD_SEPARATOR="1"
VIEWER_SEPARATOR_WIDTH="5"
START_VIEWER_IN_RUN_3DW="1"
INITIAL_IMAGE_NAME="brain_ready.jpg"
PREFER_NAMED_INITIAL_IMAGE="1"

# Bridge (for local smallbrain <-> cloud)
START_BRIDGE_IN_RUN_3DW="1"
BRIDGE_HOST="0.0.0.0"
BRIDGE_PORT="8000"
BRIDGE_PROMPT_FILE="$OUT_ROBOT_PROMPT_FILE"
BRIDGE_MONITOR_DIR="$MONITOR_DIR"
CLEAR_SHARE_PHOTO_ON_START="1"
KEEP_BRIDGE_ON_EXIT="1"

# Task instruction
USER_INSTRUCTION="grasp the blue cube"
INTERACTIVE_SELECT="0"

# Offline evaluation
EVAL_IMAGE_ROOT="/root/autodl-tmp/test_images_merged"
EVAL_MODE="3dw"        # 3dw | selector
EVAL_REPORT_JSON="/root/autodl-tmp/logs/eval_report.json"

# Pairwise prompt source (3dw)
PAIRWISE_PROMPT_JSONL=""
PAIRWISE_PROMPT_TEXT=""

# 3dw runtime behavior: first frame by bigbrain, then supervisor-only on new frames
SUPERVISOR_ONLY_AFTER_FIRST="1"
SUPERVISOR_ONLY_TIMEOUT_SEC="0"
SUPERVISOR_ONLY_MAX_IMAGES="0"
DISABLE_SUPERVISOR_LOOP="0"
SKIP_INITIAL_SUPERVISOR="0"

# Optional robot endpoint
ROBOT_URL=""
