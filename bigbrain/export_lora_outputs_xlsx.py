import argparse
import json
import os
import re
from glob import glob
from pathlib import Path

import torch
from PIL import Image
from openpyxl import Workbook
from transformers import AutoProcessor, AutoModelForVision2Seq

try:
    from peft import PeftModel
    PEFT_OK = True
except Exception:
    PEFT_OK = False


def list_images(image_dir: str):
    exts = ["png", "jpg", "jpeg", "webp", "bmp"]
    paths = []
    for e in exts:
        paths += glob(os.path.join(image_dir, f"**/*.{e}"), recursive=True)
        paths += glob(os.path.join(image_dir, f"**/*.{e.upper()}"), recursive=True)
    return sorted(set(paths))


def load_human_prompt_from_jsonl(jsonl_path: str) -> str:
    """
    从 LLaMA-Factory conversations 里抽第一条 human/user 的 value。
    例如 "<image>grasp blue cube" -> "grasp blue cube"
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = obj.get("conversations")
            if isinstance(conv, list):
                for turn in conv:
                    if not isinstance(turn, dict):
                        continue
                    frm = (turn.get("from") or "").lower()
                    if frm in ("human", "user"):
                        val = turn.get("value") or ""
                        # 去掉 <image> 占位
                        val = re.sub(r"<\s*image\s*>", " ", val, flags=re.IGNORECASE)
                        val = re.sub(r"<\s*img\s*>", " ", val, flags=re.IGNORECASE)
                        return val.strip()
    return ""


def resolve_base_model_from_adapter(lora_dir: str) -> str:
    cfg_path = os.path.join(lora_dir, "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base = cfg.get("base_model_name_or_path") or cfg.get("base_model") or ""
        if isinstance(base, str) and base.strip():
            return base.strip()
    return ""


def build_messages(prompt_text: str):    
# 重要：这里不加任何额外“只输出/不要解释”，完全贴合你训练的 human prompt 语气    
    return [        
    {            
    "role": "user",            
    "content": [                
    {"type": "image"},                
    {"type": "text", "text": prompt_text},            
    ],        
    }    
    ]


def is_lora_adapter_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def is_full_checkpoint_dir(path: str) -> bool:
    # 许多整模型 checkpoint 会有 config.json / model.safetensors 等
    return os.path.exists(os.path.join(path, "config.json"))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_or_lora_dir", type=str, required=True,
                        help="你的 checkpoint-1014 路径（可能是 LoRA adapter 或整模型 checkpoint）")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="测试图片文件夹")
    parser.add_argument("--jsonl_prompt", type=str, required=True,
                        help="用于读取 human prompt 的 grasp_final_ready.jsonl")
    parser.add_argument("--base_model", type=str, default="",
                        help="如果是 LoRA adapter 且 adapter_config.json 没写 base_model，就手动传这个")
    parser.add_argument("--out_xlsx", type=str, default="lora_outputs.xlsx")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    ckpt_dir = args.ckpt_or_lora_dir
    image_dir = args.image_dir

    # 1) prompt：从 jsonl 抽取 human 的 value（例如 grasp blue cube）
    prompt = load_human_prompt_from_jsonl(args.jsonl_prompt)
    if not prompt:
        raise RuntimeError("没能从 jsonl 里读到 human prompt，请检查 jsonl 路径/格式。")

    # 2) 图片列表
    image_paths = list_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"在 {image_dir} 没找到图片。")
    if args.limit and args.limit > 0:
        image_paths = image_paths[:args.limit]

    # 3) 判定加载方式
    use_lora = is_lora_adapter_dir(ckpt_dir)
    use_full = is_full_checkpoint_dir(ckpt_dir)

    if use_lora and not PEFT_OK:
        raise RuntimeError("检测到是 LoRA adapter 目录，但环境没有 peft。请先 pip install peft")

    if use_lora:
        base_model = args.base_model.strip() if args.base_model else ""
        if not base_model:
            base_model = resolve_base_model_from_adapter(ckpt_dir)
        if not base_model:
            raise RuntimeError("这是 LoRA adapter，但无法从 adapter_config.json 推断 base_model；请加 --base_model")
        model_path = base_model
    elif use_full:
        # 整模型 checkpoint：直接加载这个目录
        model_path = ckpt_dir
    else:
        raise RuntimeError(
            "无法判断这是 LoRA adapter 还是整模型 checkpoint。\n"
            "请检查目录里是否存在 adapter_config.json（LoRA）或 config.json（整模型）。"
        )

    print("========== CONFIG ==========")
    print("ckpt_or_lora_dir:", ckpt_dir)
    print("load_as_lora     :", use_lora)
    print("model_path       :", model_path)
    print("image_dir        :", image_dir)
    print("num_images       :", len(image_paths))
    print("human_prompt     :", prompt)
    print("============================")

    # 4) 加载 processor + model
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # 如果是 LoRA，把 adapter 挂上去
    if use_lora:
        model = PeftModel.from_pretrained(model, ckpt_dir)
        model.eval()

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # 5) Excel 输出：两列（image_name / lora_output）
    wb = Workbook()
    ws = wb.active
    ws.title = "results"
    ws.append(["image_name", "lora_output"])

    for i, img_path in enumerate(image_paths, start=1):
        img = Image.open(img_path).convert("RGB")

        messages = build_messages(prompt)

        if hasattr(processor, "apply_chat_template"):
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
        else:
            text = prompt

        inputs = processor(text=[text], images=[img], return_tensors="pt")
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(model.device)

        output_ids = model.generate(**inputs, **gen_kwargs)

        if hasattr(processor, "batch_decode"):
            decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        else:
            decoded = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 尽量裁掉 prompt（不同模板可能不完全一致，但一般可用）
        if isinstance(text, str) and text in decoded:
            decoded = decoded.split(text, 1)[-1].strip()

        ws.append([os.path.basename(img_path), decoded])

        if i % 20 == 0:
            print(f"[{i}/{len(image_paths)}] {os.path.basename(img_path)} -> {decoded}")

    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_xlsx))
    print(f"\n✅ Done. Saved to: {out_xlsx}")


if __name__ == "__main__":
    main()
