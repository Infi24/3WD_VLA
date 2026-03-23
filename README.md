# 3WD-VLA

3WD-VLA is a hierarchical dual-view robotic grasping framework for safe manipulation with vision-language reasoning. It combines a supervisor pre-check module, a Three-Way Decision core, selector-based boundary reasoning, and robot-side execution support for real-world deployment.

This repository is centered on a merged two-view observation setting:
- Left: top-view image
- Right: hand/wrist-view image

The system is designed for robotic grasp planning under ambiguity, occlusion, stacking, and safety constraints, and has been tested in a LeRobot SO100-style deployment workflow.

## Highlights

- Hierarchical decision pipeline:
  - supervisor pre-check
  - three-way decision scoring
  - selector fallback for boundary cases
- Dual-view reasoning with merged top-view and wrist-view observations
- Safety-aware control with human-hand detection, target-missing detection, and low-light handling
- Boundary-case planning for:
  - occlusion
  - stacking
  - multi-target ambiguity
- Robot-side prompt execution support through small-brain integration
- Practical deployment utilities for image receiving, bridging, logging, and runtime orchestration

## Core Idea

The project follows a three-layer collaboration design:

1. Supervisor Brain
- Performs negative-domain screening before execution
- Detects unsafe or invalid cases such as:
  - human hand in scene
  - target object missing
  - dim lighting

2. Big Brain
- Performs high-level reasoning on the merged image
- Uses pairwise preference reasoning and a Three-Way Decision score to distinguish:
  - direct execution
  - boundary-domain reasoning
  - negative-domain rejection
- Falls back to selector reasoning when the scene is not directly executable

3. Small Brain
- Reads robot prompts or action files
- Converts high-level decisions into robot-side execution behavior

## Repository Structure

```text
3WD-VLA/
  bigbrain/                 # three-way decision, selector, orchestration
  share/                    # viewer, bridge, instruction library, image utilities
  supervisor_brain/         # safety / negative-domain pre-check
  past/                     # earlier experimental versions and references
  backups/                  # snapshots / archived runner versions
  smallbrain_file_prompt.py # robot-side prompt execution script
  config.sh                 # deployment configuration
  run_3dw.sh                # 3WD runtime entry
  WORKFLOW.md               # function map and runtime chain
  README.md
```

## Main Modules

- `bigbrain/pipeline_runner.py`
  - Unified entry for `sequential`, `selector`, and `3dw` modes

- `bigbrain/3dw.py`
  - Computes the Three-Way Decision score `alpha`

- `bigbrain/selector.py`
  - Handles boundary-domain reasoning such as occlusion, stacking, and multiple targets

- `supervisor_brain/supervisor_core.py`
  - Performs safety pre-check and negative-domain judgment

- `share/share_viewer.py`
  - Receives top/hand frames and writes merged images such as `brain_ready.jpg`

- `share/bridge_http.py`
  - Bridges prompt and frame communication across local/cloud execution

- `share/instruction_library.py`
  - Maps action codes to robot prompts and UI messages

## Runtime Modes

### 1. `sequential`
- Supervisor pre-check first
- Big Brain then makes multi-round decisions until a positive action is reached

### 2. `selector`
- Supervisor pre-check first
- Runs one-shot structured reasoning for action selection

### 3. `3dw`
- Supervisor pre-check first
- Pairwise preference model generates direct / ask / replan / reject relations
- `3dw.py` computes the alpha score
- Positive-domain samples go to direct grasp
- Boundary-domain samples fall back to selector reasoning

## Three-Way Decision

The Three-Way Decision module estimates whether direct grasp is reliable enough to execute immediately.

It first generates pairwise preferences among four candidate actions:
- `direct`
- `ask`
- `replan`
- `reject`

Then it computes a latent action preference distribution and derives:

```text
p_rel = π_direct / (π_direct + π_ask)
p_aff = π_direct / (π_direct + π_replan + π_reject)
alpha = p_rel · p_aff
```

where `alpha` is the final three-way decision score.

If `alpha > threshold`, the sample is treated as positive-domain.
Otherwise, it is treated as boundary-domain and routed to selector reasoning.

## Prerequisites

- Python 3.10+
- Recommended core packages:
  - `torch`
  - `transformers`
  - `peft`
  - `pillow`
  - `requests`
  - `numpy`

Example:

```bash
pip install torch transformers peft pillow requests numpy
```

## Quick Start

### 1. Edit configuration

The main runtime configuration is currently stored in:

```bash
config.sh
```

Typical fields you may need to change:
- `MODE`
- `MONITOR_DIR`
- `CKPT_OR_LORA_DIR`
- `BASE_MODEL`
- `SUPERVISOR_MODEL_PATH`
- `ROBOT_URL`

### 2. Run the 3WD pipeline

```bash
bash run_3dw.sh
```

### 3. Or run the unified pipeline directly

```bash
python bigbrain/pipeline_runner.py \
  --mode 3dw \
  --monitor_dir /path/to/share_photo \
  --ckpt_or_lora_dir /path/to/checkpoint \
  --base_model /path/to/base_model \
  --supervisor_backend local \
  --supervisor_model_path /path/to/supervisor_model \
  --user_instruction "grasp the blue cube"
```

## Outputs

The system writes:

- merged images such as `brain_ready.jpg` and `combined_N.jpg`
- action code files
- robot prompt files
- structured JSON run logs

These outputs support both debugging and experiment recording.

## Deployment Notes

This repository includes practical components for real-world use:

- dual-camera merged observation
- cloud/local bridge communication
- prompt-based robot-side execution
- LeRobot-style SO100 deployment references

Some parts of the full system depend on:
- local hardware
- external checkpoints
- robot communication endpoints

So this repository should be understood as a reproducible research codebase plus deployment utilities, rather than a single-command plug-and-play package.

## Documentation

- [WORKFLOW.md](WORKFLOW.md): detailed runtime chain and function map
- `past/`: earlier versions and prompt-engineering references

## Open-Source Notes

Before public release, make sure to:

- remove API keys and private endpoints
- avoid committing large checkpoints and zip files
- replace machine-specific deployment paths with public templates

See also:
- [`OPEN_SOURCE_CHECKLIST.md`](OPEN_SOURCE_CHECKLIST.md)

## Acknowledgment

This project combines:
- hierarchical decision design for safe robotic grasping
- three-way decision modeling
- selector-based boundary reasoning
- deployment support for real robot execution workflows

It is intended to support both research reporting and practical robotic experiments.
