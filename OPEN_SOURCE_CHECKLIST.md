# Open Source Checklist

This repository is close to being shareable, but it is not ready to publish as-is yet.

## High-priority blockers

1. Remove secrets
- `supervisor_brain/main_supervisor.py` currently contains a hard-coded `API_KEY`.
- This must be removed before pushing to GitHub.

2. Exclude large files
- The repository currently contains very large artifacts such as:
  - `lora_output/adapter_model.safetensors`
  - `lora_output.zip`
  - `test——images.zip`
- These should not be pushed to a normal GitHub repository.

3. Separate public config from local deployment config
- `config.sh` contains machine-specific paths like `/root/autodl-tmp/...`.
- Before open-sourcing, keep a public template such as `config.example.sh` and keep personal deployment values local.

## Recommended next steps

1. Add a license
- Recommended simple choice: `MIT`
- Better for research code if you want attribution-only reuse.

2. Improve setup docs
- Add installation instructions for:
  - Python version
  - dependencies
  - model checkpoint placement
  - expected image folder layout

3. Clarify repository structure
- Explain the roles of:
  - `bigbrain/`
  - `supervisor_brain/`
  - `share/`
  - `smallbrain_file_prompt.py`
  - `past/`

4. Add a reproducibility note
- Explain which parts are open-source code only
- Explain which parts require local hardware, models, or external checkpoints

## Suggested public release sequence

1. Clean secrets and large files
2. Add `LICENSE`
3. Add `config.example.sh`
4. Update `README.md` for public users
5. Initialize Git and commit
6. Create GitHub repository
7. Push the cleaned repository

## Notes for this repository

- The strongest public-facing contribution is the hierarchical decision pipeline:
  - supervisor pre-check
  - three-way decision
  - selector fallback
  - bridge/viewer for dual-view robotic execution
- The `past/` folder may be better kept as:
  - an appendix reference in the paper, or
  - a separate `legacy/` folder with a short explanation
