import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from share.instruction_library import ActionCodes, InstructionRegistry
from bigbrain import selector
from supervisor_brain.supervisor_core import SupervisorCore, SystemState
from bigbrain.pipeline_runner import PairwiseDecisionEngine, _extract_pairwise_segment


CLASS_EXPECTED = {
    "N1": "NEGATIVE",
    "N2": "NEGATIVE",
    "P": "POSITIVE",
    "S1": "BOUNDARY",
    "S2": "BOUNDARY",
    "S3": "BOUNDARY",
}


def load_human_prompt_from_jsonl(jsonl_path: str) -> str:
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


def resolve_pairwise_prompt(args: argparse.Namespace) -> str:
    if args.pairwise_prompt_jsonl:
        return load_human_prompt_from_jsonl(args.pairwise_prompt_jsonl)
    if args.pairwise_prompt_text:
        return args.pairwise_prompt_text
    return (
        "Image shows Top-view (left) and Hand-view (right). Analyze the grasp task: "
        "\"grasp the blue cube\". Provide the pairwise decision logic followed by the "
        "safety reasoning using \"||\" as a separator."
    )


def iter_images(root: Path, limit_per_class: int = 0) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    for cls in sorted(CLASS_EXPECTED.keys()):
        cls_dir = root / cls
        if not cls_dir.exists():
            continue
        imgs = sorted([p for p in cls_dir.glob("*.jpg")] + [p for p in cls_dir.glob("*.png")])
        if limit_per_class and limit_per_class > 0:
            imgs = imgs[:limit_per_class]
        for p in imgs:
            items.append((cls, p))
    return items


def classify_from_actions(actions: List[str]) -> str:
    if any(a in (ActionCodes.REQ_USER_SELECT, ActionCodes.ACT_REMOVE_TOP, ActionCodes.ACT_CLEAR_PATH) for a in actions):
        return "BOUNDARY"
    return "POSITIVE"


def instruction_correct(cls: str, actions: List[str]) -> bool:
    if cls == "S1":
        return ActionCodes.REQ_USER_SELECT in actions
    if cls == "S2":
        return actions[:2] == [ActionCodes.ACT_CLEAR_PATH, ActionCodes.GRASP_STANDARD]
    if cls == "S3":
        return actions[:2] == [ActionCodes.ACT_REMOVE_TOP, ActionCodes.GRASP_BOTTOM]
    if cls == "P":
        return actions[:1] == [ActionCodes.GRASP_STANDARD]
    if cls in ("N1", "N2"):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation for 3WD pipeline")
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--mode", choices=["3dw", "selector"], default="3dw")
    parser.add_argument("--ckpt_or_lora_dir", type=str, default="")
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--pairwise_prompt_text", type=str, default="")
    parser.add_argument("--pairwise_prompt_jsonl", type=str, default="")
    parser.add_argument("--alpha_threshold", type=float, default=0.85)
    parser.add_argument("--user_instruction", type=str, default="grasp the blue cube")
    parser.add_argument("--limit_per_class", type=int, default=0)
    parser.add_argument("--report_json", type=str, default="eval_report.json")
    parser.add_argument("--disable_supervisor", action="store_true")
    parser.add_argument("--supervisor_backend", choices=["api", "local"], default="local")
    parser.add_argument("--supervisor_model_path", type=str, default="")
    parser.add_argument("--supervisor_lora_path", type=str, default="")
    args = parser.parse_args()

    root = Path(args.image_root)
    if not root.exists():
        raise FileNotFoundError(root)

    sup = None
    if not args.disable_supervisor:
        model_path = args.supervisor_model_path or args.base_model
        sup = SupervisorCore(
            api_key="",
            backend=args.supervisor_backend,
            local_model_path=model_path,
            local_lora_path=args.supervisor_lora_path,
        )

    pair_engine = None
    if args.mode == "3dw":
        pair_engine = PairwiseDecisionEngine(
            ckpt_or_lora_dir=args.ckpt_or_lora_dir,
            base_model=args.base_model,
        )
        prompt_text = resolve_pairwise_prompt(args)

    items = iter_images(root, args.limit_per_class)
    results = []

    correct_decision = 0
    correct_instruction = 0

    for cls, img_path in items:
        expected = CLASS_EXPECTED.get(cls, "UNKNOWN")
        record = {
            "class": cls,
            "image": str(img_path),
            "expected": expected,
        }

        # Supervisor decision (negative)
        if sup is not None:
            vision = sup.vlm_reasoning(str(img_path))
            state, _, meta = sup.decide(vision)
            record["supervisor_state"] = state.name
            record["supervisor_meta"] = meta
            if state in (SystemState.NEGATIVE_1, SystemState.NEGATIVE_2):
                record["decision"] = "NEGATIVE"
                record["actions"] = []
                record["instruction_ok"] = (expected == "NEGATIVE")
                record["decision_ok"] = (expected == "NEGATIVE")
                correct_decision += int(record["decision_ok"])
                correct_instruction += int(record["instruction_ok"])
                results.append(record)
                continue

        # Non-negative path
        if args.mode == "selector":
            def _vlm_call(img_path_str: str, prompt: str) -> str:
                if sup is None:
                    raise RuntimeError("Supervisor backend required for selector VLM call")
                return sup.call_vlm(img_path_str, prompt)

            out = selector.analyze_image_and_select(
                vlm_call=_vlm_call,
                image_path=str(img_path),
                user_instruction=args.user_instruction,
            )
            actions = out["actions"]
            decision = classify_from_actions(actions)
            record["actions"] = actions
            record["decision"] = decision
            record["decision_ok"] = (decision == expected)
            record["instruction_ok"] = instruction_correct(cls, actions)
        else:
            raw = pair_engine.infer_pairwise(str(img_path), prompt_text)
            segment = _extract_pairwise_segment(raw)
            # 3dw alpha
            from importlib.util import spec_from_file_location, module_from_spec
            mod_path = Path(__file__).resolve().parents[1] / "bigbrain" / "3dw.py"
            spec = spec_from_file_location("three_dw", str(mod_path))
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            alpha = mod.calculate_alpha_only(segment)
            if alpha is None:
                decision = "BOUNDARY"
            else:
                decision = "POSITIVE" if alpha > args.alpha_threshold else "BOUNDARY"
            record["pairwise_raw"] = raw
            record["pairwise_segment"] = segment
            record["alpha"] = alpha
            record["decision"] = decision
            record["decision_ok"] = (decision == expected)

            actions = []
            if decision == "POSITIVE":
                actions = [ActionCodes.GRASP_STANDARD]
            else:
                def _vlm_call(img_path_str: str, prompt: str) -> str:
                    if sup is None:
                        raise RuntimeError("Supervisor backend required for selector VLM call")
                    return sup.call_vlm(img_path_str, prompt)

                out = selector.analyze_image_and_select(
                    vlm_call=_vlm_call,
                    image_path=str(img_path),
                    user_instruction=args.user_instruction,
                )
                actions = out["actions"]

            record["actions"] = actions
            record["instruction_ok"] = instruction_correct(cls, actions)

        correct_decision += int(record["decision_ok"])
        correct_instruction += int(record["instruction_ok"])
        results.append(record)

    summary = {
        "total": len(results),
        "decision_accuracy": correct_decision / max(len(results), 1),
        "instruction_accuracy": correct_instruction / max(len(results), 1),
    }

    out = {
        "summary": summary,
        "results": results,
    }
    Path(args.report_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
