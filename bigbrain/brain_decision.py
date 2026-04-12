import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

try:
    from peft import PeftModel
    PEFT_OK = True
except Exception:
    PEFT_OK = False


# ----------------------------
# utils
# ----------------------------
def load_py_module(py_path: str, module_name: str = "instruction_library"):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load python module: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_base_model_from_adapter(lora_dir: str) -> str:
    cfg_path = os.path.join(lora_dir, "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base = cfg.get("base_model_name_or_path") or cfg.get("base_model") or ""
        if isinstance(base, str) and base.strip():
            return base.strip()
    return ""


def is_lora_adapter_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def build_messages_for_action_select(img_path: str, menu_text: str):
    """
    让模型只输出一个 ActionCode（必须来自菜单）
    """
    prompt = (
        "You are the Big Brain for a robot arm.\n"
        "Choose exactly ONE action code from the menu below.\n"
        "Rules:\n"
        "1) Output ONLY the action code string (one line). No extra words.\n"
        "2) The action code MUST be one of the menu items.\n"
        "3) If uncertain, choose REQ_USER_SELECT.\n\n"
        "=== ACTION MENU ===\n"
        f"{menu_text}\n"
        "=== END MENU ===\n"
    )

    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": prompt},
        ],
    }]


def extract_action_code(text: str, valid_codes: set[str]) -> str | None:
    """
    尝试从模型输出里抽出一个合法 action code
    """
    s = text.strip()

    # 直接命中
    if s in valid_codes:
        return s

    # 有些模型会多输出空格/换行/引号
    s2 = s.strip().strip('"').strip("'")
    if s2 in valid_codes:
        return s2

    # 从文本里找任意一个 valid code
    # （严格：只要找到第一个就用）
    for code in valid_codes:
        if re.search(rf"\b{re.escape(code)}\b", s):
            return code

    return None


def write_action_to_file(action_code: str, out_path: str):
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(action_code + "\n", encoding="utf-8")


# ----------------------------
# main
# ----------------------------
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--ckpt_or_lora_dir", type=str, required=True,
                        help="LoRA checkpoint dir, e.g. .../checkpoint-2700")
    parser.add_argument("--base_model", type=str, default="",
                        help="If adapter_config.json doesn't contain base model, pass it here")

    # instruction library
    parser.add_argument("--instruction_lib", type=str, required=True,
                        help="Path to your instruction_library.py")

    # image
    parser.add_argument("--image_path", type=str, default="/root/autodl-tmp/share_photo/brain_initial.jpg",
                        help="Image to analyze (initial or ready)")

    # output to cerebellum
    parser.add_argument("--out_action_file", type=str, default="/root/autodl-tmp/share_cmd/action_code.txt",
                        help="Write selected action code to this file (cerebellum can poll it)")

    # generation
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    # 1) load instruction library (python)
    lib = load_py_module(args.instruction_lib, "instruction_library")
    # 你的库里有 InstructionRegistry / ActionCodes / get_brain_menu() :contentReference[oaicite:1]{index=1}
    menu_text = lib.InstructionRegistry.get_brain_menu()
    valid_codes = set(lib.InstructionRegistry.INSTRUCTIONS.keys())

    # 2) load model + processor + lora
    ckpt = args.ckpt_or_lora_dir
    if not is_lora_adapter_dir(ckpt):
        raise RuntimeError(f"Not a LoRA adapter dir (missing adapter_config.json): {ckpt}")
    if not PEFT_OK:
        raise RuntimeError("peft not available. Please: pip install peft")

    base_model = args.base_model.strip() if args.base_model else ""
    if not base_model:
        base_model = resolve_base_model_from_adapter(ckpt)
    if not base_model:
        raise RuntimeError("Cannot resolve base model from adapter_config.json; pass --base_model")

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    model = PeftModel.from_pretrained(model, ckpt, is_trainable=False).eval()

    # 3) build prompt and run
    img = Image.open(args.image_path).convert("RGB")
    messages = build_messages_for_action_select(args.image_path, menu_text)

    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # fallback (rare)
        text = menu_text

    inputs = processor(text=[text], images=[img], return_tensors="pt")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(model.device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
    )
    out_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = out_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    action = extract_action_code(raw, valid_codes)
    if action is None:
        action = "REQ_USER_SELECT" if "REQ_USER_SELECT" in valid_codes else sorted(valid_codes)[0]

    # 4) optional: show UI + robot instruction (from library)
    inst = lib.InstructionRegistry.get_instruction(action)  # {'ui':..., 'robot':..., 'desc':...} :contentReference[oaicite:2]{index=2}

    print("========== BIGBRAIN ACTION ==========")
    print("image_path :", args.image_path)
    print("raw_output :", raw)
    print("action_code:", action)
    print("ui        :", inst.get("ui", ""))
    print("robot     :", inst.get("robot", ""))
    print("desc      :", inst.get("desc", ""))
    print("====================================")

    # 5) send to cerebellum (file)
    write_action_to_file(action, args.out_action_file)
    print(f"[OK] wrote action_code to: {args.out_action_file}")


if __name__ == "__main__":
    main()
