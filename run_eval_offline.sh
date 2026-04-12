#!/usr/bin/env bash
set -euo pipefail

# ====== Load config ======
source ./config.sh

python bigbrain/eval_offline.py \
  --image_root "$EVAL_IMAGE_ROOT" \
  --mode "$EVAL_MODE" \
  --ckpt_or_lora_dir "$CKPT_OR_LORA_DIR" \
  --base_model "$BASE_MODEL" \
  --supervisor_backend "$SUPERVISOR_BACKEND" \
  --supervisor_model_path "$SUPERVISOR_MODEL_PATH" \
  --supervisor_lora_path "$SUPERVISOR_LORA_PATH" \
  --pairwise_prompt_jsonl "$PAIRWISE_PROMPT_JSONL" \
  --pairwise_prompt_text "$PAIRWISE_PROMPT_TEXT" \
  --user_instruction "$USER_INSTRUCTION" \
  --report_json "$EVAL_REPORT_JSON"
