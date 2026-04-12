import base64
import json
import logging
import os
from enum import Enum, auto

import requests
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

try:
    from peft import PeftModel

    PEFT_OK = True
except Exception:
    PEFT_OK = False

API_BASE_URL = "https://aiping.cn/api/v1"
MODEL_NAME = "glm-4.6v"


class SystemState(Enum):
    POSITIVE = auto()
    BOUNDARY = auto()
    NEGATIVE_1 = auto()
    NEGATIVE_2 = auto()


class SupervisorCore:
    def __init__(
        self,
        api_key: str = "",
        backend: str = "api",
        local_model_path: str = "",
        local_lora_path: str = "",
        local_max_new_tokens: int = 256,
        local_temperature: float = 0.1,
        local_top_p: float = 1.0,
    ):
        self.api_key = api_key
        self.backend = backend
        self.current_state = SystemState.POSITIVE
        self.local_model_path = local_model_path
        self.local_lora_path = local_lora_path
        self.local_max_new_tokens = local_max_new_tokens
        self.local_temperature = local_temperature
        self.local_top_p = local_top_p
        self.processor = None
        self.model = None

        if self.backend not in {"api", "local"}:
            raise ValueError(f"Unsupported backend: {self.backend}")
        if self.backend == "api" and not self.api_key:
            raise ValueError("api_key is required when backend=api")
        if self.backend == "local":
            if not self.local_model_path:
                raise ValueError("local_model_path is required when backend=local")
            self._init_local_model()

    def _init_local_model(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            self.local_model_path, trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            self.local_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        if self.local_lora_path:
            if not PEFT_OK:
                raise RuntimeError("peft not available, cannot load local_lora_path")
            model = PeftModel.from_pretrained(model, self.local_lora_path).eval()
        self.model = model

    def encode_image(self, image_path: str) -> str | None:
        if not os.path.exists(image_path):
            logging.error(f"File not found: {image_path}")
            return None
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logging.error(f"Error encoding image {image_path}: {e}")
            return None

    def _extract_text(self, content: str) -> str:
        if not content:
            return ""
        return content.strip().replace("```json", "").replace("```", "").strip()

    def _call_vlm_api(self, image_path: str, prompt: str) -> str:
        b64_img = self.encode_image(image_path)
        if not b64_img:
            return json.dumps({"description": "Image load error"})

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                ],
            }],
            "temperature": 0.1,
            "max_tokens": 800,
        }

        url = f"{API_BASE_URL}/chat/completions"
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code != 200:
            logging.error(f"API Error {r.status_code}: {r.text}")
            return json.dumps({"description": f"API Error: {r.status_code}"})
        content = r.json()["choices"][0]["message"]["content"]
        return self._extract_text(content)

    @torch.inference_mode()
    def _call_vlm_local(self, image_path: str, prompt: str) -> str:
        if not self.model or not self.processor:
            raise RuntimeError("Local model is not initialized")
        if not os.path.exists(image_path):
            return json.dumps({"description": "Image load error"})

        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }]
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=self.local_max_new_tokens,
            do_sample=(self.local_temperature > 0),
            temperature=self.local_temperature,
            top_p=self.local_top_p,
        )
        out_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = out_ids[:, inputs["input_ids"].shape[1] :]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return self._extract_text(raw)

    def call_vlm(self, image_path: str, prompt: str) -> str:
        if self.backend == "local":
            return self._call_vlm_local(image_path, prompt)
        return self._call_vlm_api(image_path, prompt)

    def vlm_reasoning(self, img_path: str) -> dict:
        prompt = """
You are a safety supervisor. Analyze this MERGED image (Left: Top View, Right: Hand View).

CRITICAL INSTRUCTION:
- The TARGET OBJECT is a BLUE CUBE.
- Only detect REAL human skin/hands.

Answer 3 questions and return a JSON object with boolean results AND detailed explanations:

1. [Safety] Is a HUMAN HAND visible in EITHER view?
2. [Target] Is the target object (BLUE CUBE) MISSING in BOTH views?
   - If BLUE CUBE appears in EITHER view, target_missing must be false.
   - Only set target_missing=true when BLUE CUBE is absent in BOTH views.
3. [Light] Is the environment EXTREMELY dark?

JSON Format:
{
  "hand_detected": boolean,
  "hand_explanation": "string",
  "target_missing": boolean,
  "target_explanation": "string",
  "is_lighting_dim": boolean,
  "light_explanation": "string"
}
"""
        raw = self.call_vlm(img_path, prompt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            l = raw.find("{")
            r = raw.rfind("}")
            if l != -1 and r != -1 and r > l:
                try:
                    return json.loads(raw[l : r + 1])
                except Exception:
                    pass
            logging.error(f"JSON Parse Error. Raw: {raw}")
            return {
                "hand_detected": False,
                "hand_explanation": "Parse Error",
                "target_missing": False,
                "target_explanation": "Parse Error",
                "is_lighting_dim": True,
                "light_explanation": "Parse Error -> treat as dim",
            }

    def decide(self, vision_data: dict) -> tuple[SystemState, str | None, dict]:
        has_hand = bool(vision_data.get("hand_detected", False))
        is_missing = bool(vision_data.get("target_missing", False))
        is_dim = bool(vision_data.get("is_lighting_dim", False))

        meta = {
            "hand": has_hand,
            "missing": is_missing,
            "dim": is_dim,
            "hand_expl": vision_data.get("hand_explanation", ""),
            "target_expl": vision_data.get("target_explanation", ""),
            "light_expl": vision_data.get("light_explanation", ""),
        }

        if has_hand:
            return SystemState.NEGATIVE_1, "ALARM_TRIGGERED", meta
        if is_missing:
            return SystemState.NEGATIVE_2, None, meta
        if is_dim:
            return SystemState.BOUNDARY, "SET_SPEED_RATIO 0.3", meta
        return SystemState.POSITIVE, "SET_SPEED_RATIO 1.0", meta
