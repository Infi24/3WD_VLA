"""
Selector 模块：使用一次大模型（本地或远端）分析图片并回答三项问题，
然后基于 `share/instruction_library.InstructionRegistry` 生成相应的机器人指令和 UI 提示。

工作流程：
- 接受一个 `vlm_call` 回调（签名: func(image_path, prompt) -> str），该回调负责调用大模型并返回文本结果。
- 向大模型提出三个英文问题（要求输出 JSON）：
  1) How many blue cubes are present? (integer)
  2) Is there any occlusion that blocks direct grasp? (boolean)
  3) Is the blue cube stacked under other objects? (boolean)
- 解析 JSON 回答并根据 `InstructionRegistry` 映射决定要执行的动作序列（例如 `REQ_USER_SELECT`, `ACT_REMOVE_TOP`, `ACT_CLEAR_PATH`, `GRASP_STANDARD` 等）。
"""
from typing import Callable, Dict, Any, List
import json, logging

from share.instruction_library import InstructionRegistry, ActionCodes


def _build_prompt(instruction: str) -> str:
    return (
        "You are an assistant for robotic grasp decision. Analyze the merged image (Left: top view, Right: hand/wrist view).\n"
        "Important counting rule: Left and Right are two views of the same workspace.\n"
        "First count blue cubes in LEFT/TOP view only -> key 'blue_cube_count_left'.\n"
        "Then count blue cubes in RIGHT/HAND view only -> key 'blue_cube_count_right'.\n"
        "Then set final key 'blue_cube_count' as max(blue_cube_count_left, blue_cube_count_right).\n"
        "Answer the following three questions and return a VALID JSON object only. Do not add extra text.\n\n"
        "1) Return integer keys: 'blue_cube_count_left', 'blue_cube_count_right', and 'blue_cube_count' (the max of the first two).\n"
        "2) Is there any occlusion that would block a direct grasp? Return boolean under key 'occlusion'. "
        "Judge this using BOTH left/top and right/hand evidence.\n"
        "Define occlusion using these rules.\n"
        "Rule A: If there is an object between the gripper approach path and the blue cube, this is occlusion=true. "
        "In that case the obstacle should be removed before grasping.\n"
        "Rule B: If the blue cube is fully or mostly blocked by another object in the view, this is also strong evidence for occlusion=true.\n"
        "Rule C: An object merely next to the blue cube is weak evidence and usually should NOT be treated as occlusion unless it blocks access.\n"
        "Rule D: Use both views together. If the top view shows the target but the hand view mainly shows obstacle(s) or cannot confirm a clear path to the cube, prefer occlusion=true.\n"
        "Rule E: Do not require total invisibility of the cube. If the cube is visible but the gripper path is blocked, occlusion must still be true.\n"
        "3) Is the target blue cube stacked under other objects (i.e., another object on top of it)? Return boolean under key 'stacked'. "
        "Judge this using BOTH left/top and right/hand evidence.\n\n"
        "Additionally include an 'explanations' object with short textual reasons for each answer.\n"
        "In 'count' explanation, mention left_count, right_count, and why max(...) is used.\n\n"
        "In 'occlusion' and 'stacked' explanations, explicitly provide two clauses: one for LEFT/TOP evidence and one for RIGHT/HAND evidence.\n"
        "For 'occlusion', explicitly state whether the path from gripper to cube is clear or blocked, and name the blocking object if one exists.\n"
        "If you output occlusion=false, you must be confident that no object lies in the gripper-to-cube path.\n"
        "If RIGHT/HAND evidence is limited, say it explicitly instead of ignoring it.\n\n"
        f"User instruction: \"{instruction}\"\n"
        "JSON format example:\n"
        "{"
        "\"blue_cube_count_left\": 1,"
        "\"blue_cube_count_right\": 2,"
        "\"blue_cube_count\": 2,"
        "\"occlusion\": false,"
        "\"stacked\": false,"
        "\"explanations\": {\"count\": \"...\", \"occlusion\": \"...\", \"stacked\": \"...\"}"
        "}\n"
    )


def analyze_image_and_select(
    vlm_call: Callable[[str, str], str],
    image_path: str,
    user_instruction: str,
) -> Dict[str, Any]:
    """Use the VLM to analyze the image and return selected actions and instructions.

    Args:
        vlm_call: callable(image_path, prompt) -> str
        image_path: path to the merged image
        user_instruction: original user instruction (e.g., "grasp the blue cube")

    Returns dict with keys: `raw`, `parsed`, `actions` (list of action codes), `instructions` (robot/ui pairs)
    """
    prompt = _build_prompt(user_instruction)
    raw = vlm_call(image_path, prompt)
    parsed: Dict[str, Any] = {}

    def _to_int(v: Any, default: int = 0) -> int:
        try:
            return int(v)
        except Exception:
            return default

    def _normalize_counts(obj: Dict[str, Any]) -> None:
        has_left = "blue_cube_count_left" in obj
        has_right = "blue_cube_count_right" in obj
        if has_left or has_right:
            left = _to_int(obj.get("blue_cube_count_left", 0), 0)
            right = _to_int(obj.get("blue_cube_count_right", 0), 0)
            obj["blue_cube_count_left"] = left
            obj["blue_cube_count_right"] = right
            obj["blue_cube_count"] = max(left, right)
        else:
            obj["blue_cube_count"] = _to_int(obj.get("blue_cube_count", 1), 1)

    try:
        parsed = json.loads(raw)
        _normalize_counts(parsed)
    except Exception:
        # 尝试从非严格 JSON 中提取简单信息
        logging.warning("Selector: failed to parse JSON, attempting heuristic parsing")
        text = raw.lower()
        def _bool_contains(keys):
            return any(k in text for k in keys)
        left = 1
        right = 1
        parsed = {
            "blue_cube_count_left": left,
            "blue_cube_count_right": right,
            "blue_cube_count": max(left, right),
            "occlusion": _bool_contains([
                "occlud",
                "blocked",
                "occlusion",
                "in front of",
                "between",
                "obstacle",
                "cup blocks",
                "path is blocked",
            ]),
            "stacked": _bool_contains(["stack", "under", "below", "below another", "under another"]),
            "explanations": {"raw": raw}
        }

    actions: List[str] = []
    instructions: List[Dict[str, Any]] = []

    count = int(parsed.get("blue_cube_count", 1))
    occl = bool(parsed.get("occlusion", False))
    stacked = bool(parsed.get("stacked", False))

    # Rule 1: if multiple blue cubes and user didn't specify side, ask user to select
    if count > 1 and "left" not in user_instruction.lower() and "right" not in user_instruction.lower():
        actions.append(ActionCodes.REQ_USER_SELECT)

    # Rule 2: stacking -> remove top first
    if stacked:
        actions.append(ActionCodes.ACT_REMOVE_TOP)
        actions.append(ActionCodes.GRASP_BOTTOM)
    # Rule 3: occlusion -> clear path
    elif occl:
        actions.append(ActionCodes.ACT_CLEAR_PATH)
        actions.append(ActionCodes.GRASP_STANDARD)

    # Default: direct grasp
    if not actions:
        actions.append(ActionCodes.GRASP_STANDARD)

    # Map to robot & ui instructions
    for act in actions:
        ins = InstructionRegistry.get_instruction(act)
        instructions.append({"action_code": act, "robot": ins.get("robot"), "ui": ins.get("ui")})

    return {"raw": raw, "parsed": parsed, "actions": actions, "instructions": instructions}


if __name__ == "__main__":
    # 简要示例（需要外部提供 vlm_call）
    def fake_vlm(image, prompt):
        return json.dumps({
            "blue_cube_count": 2,
            "occlusion": False,
            "stacked": False,
            "explanations": {"count": "Two blue cubes visible"}
        })

    res = analyze_image_and_select(fake_vlm, "/tmp/dummy.jpg", "grasp the blue cube")
    print(res)
