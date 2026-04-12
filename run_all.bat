@echo off
setlocal enabledelayedexpansion

REM ====== User config ======
set MODE=selector
set MONITOR_DIR=/root/autodl-tmp/share_photo
set API_KEY=REPLACE_ME
set ROBOT_URL=
set CKPT_OR_LORA_DIR=/root/autodl-tmp/lora_output
set BASE_MODEL=/root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct
set SUPERVISOR_BACKEND=local
set SUPERVISOR_MODEL_PATH=%BASE_MODEL%
set SUPERVISOR_LORA_PATH=
set SUPERVISOR_LOOP_INTERVAL_SEC=0.5
set OUT_ACTION_FILE=/root/autodl-tmp/share_cmd/action_code.txt
set LOG_JSON=/root/autodl-tmp/logs/pipeline_result.json
set USER_INSTRUCTION=grasp the blue cube

REM Viewer config
set VIEWER_HOST=0.0.0.0
set VIEWER_PORT=9999
set VIEWER_TARGET_HEIGHT=384
set VIEWER_ADD_SEPARATOR=1
set VIEWER_SEPARATOR_WIDTH=5
set VIEWER_SAVE_DIR=%MONITOR_DIR%

REM ====== Launch viewer in a new window ======
start "3WD-Viewer" cmd /k python share\share_viewer.py

REM ====== Launch pipeline ======
if /I "%MODE%"=="sequential" (
  python bigbrain\pipeline_runner.py ^
    --mode %MODE% ^
    --monitor_dir %MONITOR_DIR% ^
    --api_key %API_KEY% ^
    --supervisor_backend %SUPERVISOR_BACKEND% ^
    --supervisor_model_path %SUPERVISOR_MODEL_PATH% ^
    --supervisor_lora_path %SUPERVISOR_LORA_PATH% ^
    --supervisor_loop_interval_sec %SUPERVISOR_LOOP_INTERVAL_SEC% ^
    --robot_url %ROBOT_URL% ^
    --out_action_file %OUT_ACTION_FILE% ^
    --log_json %LOG_JSON% ^
    --ckpt_or_lora_dir %CKPT_OR_LORA_DIR% ^
    --base_model %BASE_MODEL%
) else if /I "%MODE%"=="3dw" (
  python bigbrain\pipeline_runner.py ^
    --mode 3dw ^
    --monitor_dir %MONITOR_DIR% ^
    --api_key %API_KEY% ^
    --supervisor_backend %SUPERVISOR_BACKEND% ^
    --supervisor_model_path %SUPERVISOR_MODEL_PATH% ^
    --supervisor_lora_path %SUPERVISOR_LORA_PATH% ^
    --supervisor_loop_interval_sec %SUPERVISOR_LOOP_INTERVAL_SEC% ^
    --robot_url %ROBOT_URL% ^
    --out_action_file %OUT_ACTION_FILE% ^
    --log_json %LOG_JSON% ^
    --ckpt_or_lora_dir %CKPT_OR_LORA_DIR% ^
    --base_model %BASE_MODEL% ^
    --user_instruction "%USER_INSTRUCTION%"
) else (
  python bigbrain\pipeline_runner.py ^
    --mode %MODE% ^
    --monitor_dir %MONITOR_DIR% ^
    --api_key %API_KEY% ^
    --supervisor_backend %SUPERVISOR_BACKEND% ^
    --supervisor_model_path %SUPERVISOR_MODEL_PATH% ^
    --supervisor_lora_path %SUPERVISOR_LORA_PATH% ^
    --supervisor_loop_interval_sec %SUPERVISOR_LOOP_INTERVAL_SEC% ^
    --robot_url %ROBOT_URL% ^
    --out_action_file %OUT_ACTION_FILE% ^
    --log_json %LOG_JSON% ^
    --user_instruction "%USER_INSTRUCTION%"
)

endlocal
