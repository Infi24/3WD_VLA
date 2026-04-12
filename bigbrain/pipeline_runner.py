import argparse
import json
import os
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from share.instruction_library import ActionCodes, InstructionRegistry
from bigbrain import selector
import importlib.util
import re
from bigbrain.brain_decision import (
    build_messages_for_action_select,
    extract_action_code,
    is_lora_adapter_dir,
    resolve_base_model_from_adapter,
)
from supervisor_brain.robot_client import RobotClient
from supervisor_brain.supervisor_core import SupervisorCore, SystemState

try:
    from peft import PeftModel

    PEFT_OK = True
except Exception:
    PEFT_OK = False


POSITIVE_ACTIONS = {
    ActionCodes.GRASP_STANDARD,
    ActionCodes.GRASP_LEFT_TARGET,
    ActionCodes.GRASP_RIGHT_TARGET,
    ActionCodes.GRASP_BOTTOM,
    ActionCodes.GRASP_VERTICAL,
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _image_meta(image_path: str) -> Dict[str, Any]:
    p = Path(image_path)
    if not p.exists():
        return {"path": image_path, "exists": False}
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": int(st.st_size),
        "mtime": float(st.st_mtime),
    }


def _trace(result: Dict[str, Any], stage: str, **kwargs) -> None:
    trace = result.setdefault("trace", [])
    item = {"ts": _iso_now(), "stage": stage}
    item.update(kwargs)
    trace.append(item)


def latest_image(path: str) -> Optional[Path]:
    p = Path(path)
    if not p.exists():
        return None
    files: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(p.glob(ext))
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_mtime)


def wait_for_latest_image(monitor_dir: str, timeout_sec: int) -> Path:
    t0 = time.time()
    while True:
        lf = latest_image(monitor_dir)
        if lf is not None:
            return lf
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"No image found in {monitor_dir} within {timeout_sec}s")
        time.sleep(0.2)


def wait_for_named_image(monitor_dir: str, image_name: str, timeout_sec: int) -> Path:
    t0 = time.time()
    target = Path(monitor_dir) / image_name
    while True:
        if target.exists():
            return target
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(
                f"No named image found: {target} within {timeout_sec}s"
            )
        time.sleep(0.2)


def write_action(action_code: str, out_file: str) -> None:
    p = Path(out_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(action_code + "\n", encoding="utf-8")

def _resolve_action_seq_file(args: argparse.Namespace) -> str:
    if args.out_action_seq_file:
        return args.out_action_seq_file
    return str(Path(args.out_action_file).parent / "action_code_seq.txt")


def _resolve_robot_prompt_file(args: argparse.Namespace) -> str:
    if args.out_robot_prompt_file:
        return args.out_robot_prompt_file
    return str(Path(args.out_action_file).parent / "robot_prompt.txt")


def _init_trace_files(args: argparse.Namespace) -> None:
    for fp in (_resolve_action_seq_file(args), _resolve_robot_prompt_file(args)):
        p = Path(fp)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")


def _append_line(text: str, file_path: str) -> None:
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n")


def record_action_event(
    args: argparse.Namespace,
    action_code: str,
    robot_prompt: Optional[str] = None,
) -> None:
    _append_line(action_code, _resolve_action_seq_file(args))
    rp = robot_prompt
    if rp is None:
        inst = InstructionRegistry.get_instruction(action_code)
        rp = inst.get("robot")
    if rp:
        _append_line(rp, _resolve_robot_prompt_file(args))


class BigBrainDecisionEngine:
    # 中文注释：将 LoRA + 基座模型加载一次，循环中重复推理，避免每轮重复初始化。
    def __init__(
        self,
        ckpt_or_lora_dir: str,
        instruction_lib_py: str,
        base_model: str = "",
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.ckpt_or_lora_dir = ckpt_or_lora_dir
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.valid_codes = set(InstructionRegistry.INSTRUCTIONS.keys())
        self.menu_text = InstructionRegistry.get_brain_menu()
        self.processor, self.model = self._load_model()
        self.instruction_lib_py = instruction_lib_py

    def _load_model(self):
        if not is_lora_adapter_dir(self.ckpt_or_lora_dir):
            raise RuntimeError(
                f"ckpt_or_lora_dir is not a LoRA adapter dir: {self.ckpt_or_lora_dir}"
            )
        if not PEFT_OK:
            raise RuntimeError("peft not available, please install peft")

        base_model = self.base_model.strip() if self.base_model else ""
        if not base_model:
            base_model = resolve_base_model_from_adapter(self.ckpt_or_lora_dir)
        if not base_model:
            raise RuntimeError("Cannot resolve base model, please set --base_model")

        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        model = PeftModel.from_pretrained(
            model, self.ckpt_or_lora_dir, is_trainable=False
        ).eval()
        return processor, model

    @torch.inference_mode()
    def infer_action(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = Image.open(image_path).convert("RGB")
        messages = build_messages_for_action_select(image_path, self.menu_text)

        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = self.menu_text

        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
            top_p=self.top_p,
        )
        out_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = out_ids[:, inputs["input_ids"].shape[1] :]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        action_code = extract_action_code(raw, self.valid_codes)
        if action_code is None:
            action_code = ActionCodes.REQ_USER_SELECT
        inst = InstructionRegistry.get_instruction(action_code)

        return {
            "raw": raw,
            "action_code": action_code,
            "robot": inst.get("robot"),
            "ui": inst.get("ui"),
        }


class PairwiseDecisionEngine:
    # Runs LoRA model to output pairwise comparisons (direct/ask/replan/reject).
    def __init__(
        self,
        ckpt_or_lora_dir: str,
        base_model: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.ckpt_or_lora_dir = ckpt_or_lora_dir
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.processor, self.model = self._load_model()

    def _load_model(self):
        if not is_lora_adapter_dir(self.ckpt_or_lora_dir):
            raise RuntimeError(
                f"ckpt_or_lora_dir is not a LoRA adapter dir: {self.ckpt_or_lora_dir}"
            )
        if not PEFT_OK:
            raise RuntimeError("peft not available, please install peft")

        base_model = self.base_model.strip() if self.base_model else ""
        if not base_model:
            base_model = resolve_base_model_from_adapter(self.ckpt_or_lora_dir)
        if not base_model:
            raise RuntimeError("Cannot resolve base model, please set --base_model")

        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        model = PeftModel.from_pretrained(model, self.ckpt_or_lora_dir, is_trainable=False).eval()
        return processor, model

    @torch.inference_mode()
    def infer_pairwise(self, image_path: str, prompt_text: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = Image.open(image_path).convert("RGB")

        messages = _build_pairwise_messages(prompt_text)
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        else:
            text = prompt_text

        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
            top_p=self.top_p,
        )
        out_ids = self.model.generate(**inputs, **gen_kwargs)
        decoded = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]

        # Attempt to strip prompt echo
        if isinstance(text, str) and text in decoded:
            decoded = decoded.split(text, 1)[-1].strip()
        return decoded.strip()


def maybe_send_robot(
    robot_client: Optional[RobotClient], cmd: Optional[str], meta: Dict[str, Any], dry_run: bool
) -> Optional[Dict[str, Any]]:
    if not cmd:
        return None
    if dry_run or robot_client is None:
        return {"dry_run": True, "cmd": cmd, "meta": meta}
    return robot_client.send(cmd, meta=meta)


def start_supervisor_loop(
    sup: SupervisorCore,
    monitor_dir: str,
    robot: Optional[RobotClient],
    dry_run: bool,
    interval_sec: float,
    stop_event: threading.Event,
):
    # Background safety loop: keep adjusting speed or stopping on hand detection.
    last_mtime = 0.0
    while not stop_event.is_set():
        try:
            lf = latest_image(monitor_dir)
            if lf is None:
                time.sleep(interval_sec)
                continue
            mtime = lf.stat().st_mtime
            if mtime > last_mtime:
                vision = sup.vlm_reasoning(str(lf))
                state, cmd, meta = sup.decide(vision)
                if cmd:
                    maybe_send_robot(robot, cmd, meta, dry_run)
                # If hand detected, keep issuing stop but do not kill main loop.
                last_mtime = mtime
        except Exception:
            pass
        time.sleep(interval_sec)


def run_supervisor_followup_only(
    sup: SupervisorCore,
    monitor_dir: str,
    robot: Optional[RobotClient],
    dry_run: bool,
    interval_sec: float,
    start_mtime: float,
    timeout_sec: float,
    max_images: int,
) -> Dict[str, Any]:
    # After first brain decision, only supervisor checks new incoming frames.
    last_mtime = start_mtime
    started = time.time()
    processed = 0
    last_state = None
    stop_reason = "completed"

    while True:
        if timeout_sec > 0 and (time.time() - started) >= timeout_sec:
            stop_reason = "timeout"
            break
        if max_images > 0 and processed >= max_images:
            stop_reason = "max_images"
            break

        lf = latest_image(monitor_dir)
        if lf is None:
            time.sleep(interval_sec)
            continue

        mtime = lf.stat().st_mtime
        if mtime > last_mtime:
            vision = sup.vlm_reasoning(str(lf))
            state, cmd, meta = sup.decide(vision)
            last_state = state.name
            if cmd:
                maybe_send_robot(robot, cmd, meta, dry_run)
            processed += 1
            last_mtime = mtime

        time.sleep(interval_sec)

    return {
        "processed_images": processed,
        "stop_reason": stop_reason,
        "last_supervisor_state": last_state,
    }


def run_supervisor_once(core: SupervisorCore, image_path: str) -> Tuple[SystemState, Optional[str], Dict[str, Any]]:
    vision = core.vlm_reasoning(image_path)
    return core.decide(vision)


def build_supervisor(args: argparse.Namespace) -> SupervisorCore:
    model_path = args.supervisor_model_path or args.base_model
    return SupervisorCore(
        api_key=args.api_key,
        backend=args.supervisor_backend,
        local_model_path=model_path,
        local_lora_path=args.supervisor_lora_path,
        local_max_new_tokens=args.supervisor_max_new_tokens,
        local_temperature=args.supervisor_temperature,
        local_top_p=args.supervisor_top_p,
    )


def run_sequential(args: argparse.Namespace) -> Dict[str, Any]:
    # 中文注释：监督脑先做一次负域过滤，再进入序贯三支循环。
    image_path = str(wait_for_latest_image(args.monitor_dir, args.wait_image_timeout_sec))
    sup = build_supervisor(args)
    robot = RobotClient(args.robot_url, timeout=args.robot_timeout_sec) if args.robot_url else None

    initial_state, initial_cmd, initial_meta = run_supervisor_once(sup, image_path)
    result: Dict[str, Any] = {
        "mode": "sequential",
        "initial_image": image_path,
        "initial_supervisor_state": initial_state.name,
        "initial_supervisor_meta": initial_meta,
        "rounds": [],
        "final_action": None,
    }

    if initial_state in {SystemState.NEGATIVE_1, SystemState.NEGATIVE_2}:
        maybe_send_robot(robot, initial_cmd, initial_meta, args.dry_run)
        final_action = (
            ActionCodes.EMERGENCY_STOP
            if initial_state == SystemState.NEGATIVE_1
            else ActionCodes.ABORT_TARGET_MISSING
        )
        write_action(final_action, args.out_action_file)
        record_action_event(args, final_action)
        result["final_action"] = final_action
        result["status"] = "stopped_by_supervisor"
        return result

    if initial_cmd:
        maybe_send_robot(robot, initial_cmd, initial_meta, args.dry_run)

    engine = BigBrainDecisionEngine(
        ckpt_or_lora_dir=args.ckpt_or_lora_dir,
        instruction_lib_py=args.instruction_lib,
        base_model=args.base_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    last_supervisor_ts = time.time()
    for i in range(1, args.max_rounds + 1):
        current_image = str(wait_for_latest_image(args.monitor_dir, args.wait_image_timeout_sec))
        round_item: Dict[str, Any] = {"round": i, "image": current_image}

        if time.time() - last_supervisor_ts >= args.supervisor_interval_sec:
            s_state, s_cmd, s_meta = run_supervisor_once(sup, current_image)
            round_item["supervisor_state"] = s_state.name
            round_item["supervisor_meta"] = s_meta
            if s_cmd:
                maybe_send_robot(robot, s_cmd, s_meta, args.dry_run)
            last_supervisor_ts = time.time()
            if s_state in {SystemState.NEGATIVE_1, SystemState.NEGATIVE_2}:
                final_action = (
                    ActionCodes.EMERGENCY_STOP
                    if s_state == SystemState.NEGATIVE_1
                    else ActionCodes.ABORT_TARGET_MISSING
                )
                write_action(final_action, args.out_action_file)
                record_action_event(args, final_action)
                round_item["action_code"] = final_action
                round_item["reason"] = "stopped_by_periodic_supervisor"
                result["rounds"].append(round_item)
                result["final_action"] = final_action
                result["status"] = "stopped_by_supervisor"
                return result

        bb = engine.infer_action(current_image)
        action = bb["action_code"]
        write_action(action, args.out_action_file)
        record_action_event(args, action, bb.get("robot"))

        round_item["bigbrain_raw"] = bb["raw"]
        round_item["action_code"] = action
        round_item["robot"] = bb["robot"]
        round_item["ui"] = bb["ui"]
        result["rounds"].append(round_item)

        if bb["robot"]:
            maybe_send_robot(robot, bb["robot"], {"source": "bigbrain", "round": i}, args.dry_run)

        if action in POSITIVE_ACTIONS:
            result["final_action"] = action
            result["status"] = "success_positive"
            return result

        time.sleep(args.step_sleep_sec)

    result["final_action"] = ActionCodes.REQ_USER_SELECT
    result["status"] = "max_rounds_reached"
    return result


def run_selector(args: argparse.Namespace) -> Dict[str, Any]:
    # 中文注释：selector 版本只分析 initial 图像一次，直接给动作序列。
    image_path = str(wait_for_latest_image(args.monitor_dir, args.wait_image_timeout_sec))
    sup = build_supervisor(args)
    robot = RobotClient(args.robot_url, timeout=args.robot_timeout_sec) if args.robot_url else None
    stop_event = threading.Event()
    if not args.disable_supervisor_loop:
        sup_thread = threading.Thread(
            target=start_supervisor_loop,
            args=(sup, args.monitor_dir, robot, args.dry_run, args.supervisor_loop_interval_sec, stop_event),
            daemon=True,
        )
        sup_thread.start()

    initial_state, initial_cmd, initial_meta = run_supervisor_once(sup, image_path)
    result: Dict[str, Any] = {
        "mode": "selector",
        "initial_image": image_path,
        "initial_supervisor_state": initial_state.name,
        "initial_supervisor_meta": initial_meta,
        "actions": [],
    }

    if initial_state in {SystemState.NEGATIVE_1, SystemState.NEGATIVE_2}:
        maybe_send_robot(robot, initial_cmd, initial_meta, args.dry_run)
        final_action = (
            ActionCodes.EMERGENCY_STOP
            if initial_state == SystemState.NEGATIVE_1
            else ActionCodes.ABORT_TARGET_MISSING
        )
        write_action(final_action, args.out_action_file)
        record_action_event(args, final_action)
        result["final_action"] = final_action
        result["status"] = "stopped_by_supervisor"
        return result

    if initial_cmd:
        maybe_send_robot(robot, initial_cmd, initial_meta, args.dry_run)

    def _vlm_call(img_path: str, prompt: str) -> str:
        return sup.call_vlm(img_path, prompt)

    out = selector.analyze_image_and_select(
        vlm_call=_vlm_call,
        image_path=image_path,
        user_instruction=args.user_instruction,
    )
    result["selector_raw"] = out["raw"]
    result["selector_parsed"] = out["parsed"]
    result["actions"] = out["actions"]
    result["instructions"] = out["instructions"]

    for idx, ins in enumerate(out["instructions"], start=1):
        action = ins["action_code"]
        write_action(action, args.out_action_file)
        record_action_event(args, action, ins.get("robot"))
        maybe_send_robot(
            robot,
            ins.get("robot"),
            {"source": "selector", "step": idx, "action_code": action},
            args.dry_run,
        )
        time.sleep(args.step_sleep_sec)

    if args.interactive_select and ActionCodes.REQ_USER_SELECT in out["actions"]:
        resolved = _resolve_user_selection()
        write_action(resolved, args.out_action_file)
        inst = InstructionRegistry.get_instruction(resolved)
        record_action_event(args, resolved, inst.get("robot"))
        maybe_send_robot(
            robot,
            inst.get("robot"),
            {"source": "selector", "step": "resolve", "action_code": resolved},
            args.dry_run,
        )
        result["resolved_action"] = resolved
        result["resolved_instruction"] = {
            "action_code": resolved,
            "robot": inst.get("robot"),
            "ui": inst.get("ui"),
        }
        result["actions"].append(resolved)
        result["instructions"].append(result["resolved_instruction"])

    if "resolved_action" in result:
        result["final_action"] = result["resolved_action"]
    else:
        result["final_action"] = out["actions"][-1] if out["actions"] else ActionCodes.GRASP_STANDARD
    result["status"] = "done"
    stop_event.set()
    return result


def _resolve_pairwise_prompt(args: argparse.Namespace) -> str:
    if args.pairwise_prompt_jsonl:
        return _load_human_prompt_from_jsonl(args.pairwise_prompt_jsonl)
    if args.pairwise_prompt_text:
        return args.pairwise_prompt_text
    return (
        "Image shows Top-view (left) and Hand-view (right). Analyze the grasp task: "
        "\"grasp the blue cube\". Provide the pairwise decision logic followed by the "
        "safety reasoning using \"||\" as a separator."
    )


def _load_3dw_module() -> object:
    # Load bigbrain/3dw.py dynamically (module name starts with digit)
    path = ROOT / "bigbrain" / "3dw.py"
    spec = importlib.util.spec_from_file_location("three_dw", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load 3dw.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_pairwise_messages(prompt_text: str):
    # Match export_lora_outputs_xlsx prompt template (no extra instruction wrapper)
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ],
    }]


def _load_human_prompt_from_jsonl(jsonl_path: str) -> str:
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
                        val = re.sub(r"<\s*image\s*>", " ", val, flags=re.IGNORECASE)
                        val = re.sub(r"<\s*img\s*>", " ", val, flags=re.IGNORECASE)
                        return val.strip()
    return ""


def _map_positive_action(args: argparse.Namespace) -> str:
    if args.positive_action_code:
        return args.positive_action_code
    return ActionCodes.GRASP_STANDARD


def _extract_pairwise_segment(raw_text: str) -> str:
    # Only use the pairwise segment before "||"
    head = raw_text.split("||", 1)[0]
    # Extract valid comparisons to avoid safety text contamination
    pairs = re.findall(
        r"\b(direct|ask|replan|reject)\s*>\s*(direct|ask|replan|reject)\b",
        head.lower(),
    )
    if not pairs:
        return ""
    return " ".join([f"{a}>{b}" for a, b in pairs])


def _resolve_user_selection() -> str:
    # Terminal interaction for ambiguity resolution
    while True:
        ans = input("Ambiguity detected. Select target [left/right]: ").strip().lower()
        if ans in {"left", "l"}:
            return ActionCodes.GRASP_LEFT_TARGET
        if ans in {"right", "r"}:
            return ActionCodes.GRASP_RIGHT_TARGET
        print("Invalid input. Please type 'left' or 'right'.")


def run_3dw(args: argparse.Namespace) -> Dict[str, Any]:
    if args.prefer_named_initial_image:
        try:
            image_path = str(
                wait_for_named_image(
                    args.monitor_dir,
                    args.initial_image_name,
                    args.wait_image_timeout_sec,
                )
            )
        except TimeoutError:
            # Fallback: if named initial image is missing, still continue with latest image.
            image_path = str(wait_for_latest_image(args.monitor_dir, args.wait_image_timeout_sec))
    else:
        image_path = str(wait_for_latest_image(args.monitor_dir, args.wait_image_timeout_sec))

    initial_mtime = Path(image_path).stat().st_mtime
    sup = build_supervisor(args)
    robot = RobotClient(args.robot_url, timeout=args.robot_timeout_sec) if args.robot_url else None
    stop_event = threading.Event()
    if (not args.supervisor_only_after_first) and (not args.disable_supervisor_loop):
        sup_thread = threading.Thread(
            target=start_supervisor_loop,
            args=(sup, args.monitor_dir, robot, args.dry_run, args.supervisor_loop_interval_sec, stop_event),
            daemon=True,
        )
        sup_thread.start()

    result: Dict[str, Any] = {
        "mode": "3dw",
        "initial_image": image_path,
        "initial_supervisor_state": "SKIPPED" if args.skip_initial_supervisor else None,
        "initial_supervisor_meta": {} if args.skip_initial_supervisor else None,
        "debug": {
            "run_started_at": _iso_now(),
            "alpha_threshold": args.alpha_threshold,
            "pairwise_prompt_jsonl": args.pairwise_prompt_jsonl,
            "pairwise_prompt_text_provided": bool(args.pairwise_prompt_text),
            "supervisor_backend": args.supervisor_backend,
            "supervisor_only_after_first": bool(args.supervisor_only_after_first),
            "disable_supervisor_loop": bool(args.disable_supervisor_loop),
            "skip_initial_supervisor": bool(args.skip_initial_supervisor),
        },
    }
    _trace(result, "run_3dw_start", image=_image_meta(image_path))
    if not args.skip_initial_supervisor:
        initial_state, initial_cmd, initial_meta = run_supervisor_once(sup, image_path)
        result["initial_supervisor_state"] = initial_state.name
        result["initial_supervisor_meta"] = initial_meta
        _trace(
            result,
            "initial_supervisor",
            state=initial_state.name,
            cmd=initial_cmd,
            meta=initial_meta,
        )

        if initial_state in {SystemState.NEGATIVE_1, SystemState.NEGATIVE_2}:
            maybe_send_robot(robot, initial_cmd, initial_meta, args.dry_run)
            final_action = (
                ActionCodes.EMERGENCY_STOP
                if initial_state == SystemState.NEGATIVE_1
                else ActionCodes.ABORT_TARGET_MISSING
            )
            write_action(final_action, args.out_action_file)
            record_action_event(args, final_action)
            result["final_action"] = final_action
            result["status"] = "stopped_by_supervisor"
            _trace(
                result,
                "finalize",
                reason="stopped_by_supervisor",
                final_action=final_action,
            )
            stop_event.set()
            return result

        if initial_cmd:
            maybe_send_robot(robot, initial_cmd, initial_meta, args.dry_run)

    prompt_text = _resolve_pairwise_prompt(args)
    result["debug"]["pairwise_prompt_used"] = prompt_text
    _trace(
        result,
        "pairwise_prompt_resolved",
        prompt_len=len(prompt_text),
        prompt_preview=prompt_text[:300],
    )
    pair_engine = PairwiseDecisionEngine(
        ckpt_or_lora_dir=args.ckpt_or_lora_dir,
        base_model=args.base_model,
        max_new_tokens=args.pairwise_max_new_tokens,
        temperature=args.pairwise_temperature,
        top_p=args.pairwise_top_p,
    )
    raw = pair_engine.infer_pairwise(image_path, prompt_text)
    result["pairwise_raw"] = raw
    _trace(result, "pairwise_infer_done", raw_preview=raw[:500], raw_len=len(raw))

    three_dw = _load_3dw_module()
    pairwise_segment = _extract_pairwise_segment(raw)
    result["pairwise_segment"] = pairwise_segment
    alpha = three_dw.calculate_alpha_only(pairwise_segment)
    result["alpha"] = alpha
    _trace(
        result,
        "alpha_computed",
        pairwise_segment=pairwise_segment,
        alpha=alpha,
    )

    if alpha is None:
        decision = "BOUNDARY"
    else:
        decision = "POSITIVE" if alpha > args.alpha_threshold else "BOUNDARY"
    result["decision"] = decision
    _trace(
        result,
        "decision_made",
        decision=decision,
        alpha=alpha,
        alpha_threshold=args.alpha_threshold,
    )

    if decision == "POSITIVE":
        action = _map_positive_action(args)
        write_action(action, args.out_action_file)
        result["final_action"] = action
        result["status"] = "positive_direct"
        _trace(
            result,
            "positive_direct",
            mapped_action=action,
            positive_action_code_arg=args.positive_action_code,
        )
        if action and action in InstructionRegistry.INSTRUCTIONS:
            inst = InstructionRegistry.get_instruction(action)
            result["positive_robot"] = inst.get("robot")
            result["positive_ui"] = inst.get("ui")
            record_action_event(args, action, inst.get("robot"))
            if inst.get("robot"):
                maybe_send_robot(robot, inst.get("robot"), {"source": "3dw"}, args.dry_run)
        if args.supervisor_only_after_first and (not args.disable_supervisor_loop):
            result["supervisor_followup"] = run_supervisor_followup_only(
                sup=sup,
                monitor_dir=args.monitor_dir,
                robot=robot,
                dry_run=args.dry_run,
                interval_sec=args.supervisor_loop_interval_sec,
                start_mtime=initial_mtime,
                timeout_sec=args.supervisor_only_timeout_sec,
                max_images=args.supervisor_only_max_images,
            )
            _trace(result, "supervisor_followup_done", followup=result["supervisor_followup"])
        stop_event.set()
        _trace(result, "finalize", reason="positive_direct", final_action=action)
        return result

    # Boundary -> selector
    def _vlm_call(img_path: str, prompt: str) -> str:
        return sup.call_vlm(img_path, prompt)

    out = selector.analyze_image_and_select(
        vlm_call=_vlm_call,
        image_path=image_path,
        user_instruction=args.user_instruction,
    )
    _trace(result, "selector_invoked", image=_image_meta(image_path), user_instruction=args.user_instruction)
    result["selector_raw"] = out["raw"]
    result["selector_parsed"] = out["parsed"]
    result["actions"] = out["actions"]
    result["instructions"] = out["instructions"]
    _trace(
        result,
        "selector_done",
        actions=out["actions"],
        parsed=out["parsed"],
        raw_preview=str(out["raw"])[:500],
    )

    for idx, ins in enumerate(out["instructions"], start=1):
        action = ins["action_code"]
        write_action(action, args.out_action_file)
        record_action_event(args, action, ins.get("robot"))
        maybe_send_robot(
            robot,
            ins.get("robot"),
            {"source": "3dw_selector", "step": idx, "action_code": action},
            args.dry_run,
        )
        time.sleep(args.step_sleep_sec)

    if args.interactive_select and ActionCodes.REQ_USER_SELECT in out["actions"]:
        resolved = _resolve_user_selection()
        write_action(resolved, args.out_action_file)
        inst = InstructionRegistry.get_instruction(resolved)
        record_action_event(args, resolved, inst.get("robot"))
        maybe_send_robot(
            robot,
            inst.get("robot"),
            {"source": "3dw_selector", "step": "resolve", "action_code": resolved},
            args.dry_run,
        )
        result["resolved_action"] = resolved
        result["resolved_instruction"] = {
            "action_code": resolved,
            "robot": inst.get("robot"),
            "ui": inst.get("ui"),
        }
        result["actions"].append(resolved)
        result["instructions"].append(result["resolved_instruction"])

    if "resolved_action" in result:
        result["final_action"] = result["resolved_action"]
    else:
        result["final_action"] = out["actions"][-1] if out["actions"] else ActionCodes.GRASP_STANDARD
    result["status"] = "boundary_selector"
    if args.supervisor_only_after_first and (not args.disable_supervisor_loop):
        result["supervisor_followup"] = run_supervisor_followup_only(
            sup=sup,
            monitor_dir=args.monitor_dir,
            robot=robot,
            dry_run=args.dry_run,
            interval_sec=args.supervisor_loop_interval_sec,
            start_mtime=initial_mtime,
            timeout_sec=args.supervisor_only_timeout_sec,
            max_images=args.supervisor_only_max_images,
        )
        _trace(result, "supervisor_followup_done", followup=result["supervisor_followup"])
    stop_event.set()
    _trace(
        result,
        "finalize",
        reason="boundary_selector",
        final_action=result["final_action"],
    )
    return result

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified 3WD pipeline runner")
    parser.add_argument("--mode", choices=["sequential", "selector", "3dw"], required=True)
    parser.add_argument("--monitor_dir", type=str, default="./share_photo")
    parser.add_argument("--wait_image_timeout_sec", type=int, default=20)
    parser.add_argument("--prefer_named_initial_image", action="store_true")
    parser.add_argument("--initial_image_name", type=str, default="brain_ready.jpg")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--supervisor_backend", choices=["api", "local"], default="local")
    parser.add_argument("--supervisor_model_path", type=str, default="")
    parser.add_argument("--supervisor_lora_path", type=str, default="")
    parser.add_argument("--supervisor_max_new_tokens", type=int, default=256)
    parser.add_argument("--supervisor_temperature", type=float, default=0.1)
    parser.add_argument("--supervisor_top_p", type=float, default=1.0)
    parser.add_argument("--robot_url", type=str, default="")
    parser.add_argument("--robot_timeout_sec", type=float, default=2.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--out_action_file", type=str, default="./share_cmd/action_code.txt")
    parser.add_argument("--out_action_seq_file", type=str, default="")
    parser.add_argument("--out_robot_prompt_file", type=str, default="")
    parser.add_argument("--log_json", type=str, default="./logs/pipeline_result.json")
    parser.add_argument("--step_sleep_sec", type=float, default=1.0)

    # sequential specific
    parser.add_argument("--ckpt_or_lora_dir", type=str, default="./lora_output")
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--instruction_lib", type=str, default="./share/instruction_library.py")
    parser.add_argument("--max_rounds", type=int, default=6)
    parser.add_argument("--supervisor_interval_sec", type=float, default=5.0)
    parser.add_argument("--supervisor_loop_interval_sec", type=float, default=0.5)
    parser.add_argument("--disable_supervisor_loop", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    # selector specific
    parser.add_argument("--user_instruction", type=str, default="grasp the blue cube")
    parser.add_argument("--interactive_select", action="store_true")

    # 3dw / pairwise specific
    parser.add_argument("--pairwise_prompt_text", type=str, default="")
    parser.add_argument("--pairwise_prompt_jsonl", type=str, default="")
    parser.add_argument("--pairwise_max_new_tokens", type=int, default=128)
    parser.add_argument("--pairwise_temperature", type=float, default=0.0)
    parser.add_argument("--pairwise_top_p", type=float, default=1.0)
    parser.add_argument("--alpha_threshold", type=float, default=0.85)
    parser.add_argument("--positive_action_code", type=str, default="")
    parser.add_argument("--supervisor_only_after_first", action="store_true")
    parser.add_argument("--supervisor_only_timeout_sec", type=float, default=0.0)
    parser.add_argument("--supervisor_only_max_images", type=int, default=0)
    parser.add_argument("--skip_initial_supervisor", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    _init_trace_files(args)
    if args.mode == "sequential":
        res = run_sequential(args)
    elif args.mode == "selector":
        res = run_selector(args)
    else:
        res = run_3dw(args)

    res["out_action_file"] = args.out_action_file
    res["out_action_seq_file"] = _resolve_action_seq_file(args)
    res["out_robot_prompt_file"] = _resolve_robot_prompt_file(args)
    res["run_meta"] = {
        "ended_at": _iso_now(),
        "mode": args.mode,
        "monitor_dir": args.monitor_dir,
        "wait_image_timeout_sec": args.wait_image_timeout_sec,
        "ckpt_or_lora_dir": args.ckpt_or_lora_dir,
        "base_model": args.base_model,
        "supervisor_backend": args.supervisor_backend,
        "supervisor_model_path": args.supervisor_model_path,
        "pairwise_prompt_jsonl": args.pairwise_prompt_jsonl,
        "pairwise_prompt_text_provided": bool(args.pairwise_prompt_text),
        "alpha_threshold": getattr(args, "alpha_threshold", None),
    }

    log_path = Path(args.log_json)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[OK] result saved to {log_path}")


if __name__ == "__main__":
    main()
