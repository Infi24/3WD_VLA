"""
[Protocol Layer] Global Instruction Registry V3.1 (English UI & Simplified Logic)
Compatible with BigBrain (Reasoning) and Robot Controller (Execution).
"""

class ActionCodes:
    # === Priority 0: Safety (Highest) ===
    EMERGENCY_STOP = "EMERGENCY_STOP"
    
    # === Priority 1: Feasibility ===
    ABORT_TARGET_MISSING = "ABORT_TARGET_MISSING"
    ABORT_OUT_OF_BOUNDS = "ABORT_OUT_OF_BOUNDS"

    # === Priority 2: Boundary Scenarios (Complex/Ambiguous) ===
    REQ_USER_SELECT = "REQ_USER_SELECT"   # Visual Ambiguity
    ACT_REMOVE_TOP = "ACT_REMOVE_TOP"     # Stacked Objects
    ACT_CLEAR_PATH = "ACT_CLEAR_PATH"     # Occlusion (Blocked Path)

    
    # Ambiguity Resolved
    GRASP_LEFT_TARGET = "GRASP_LEFT_TARGET"
    GRASP_RIGHT_TARGET = "GRASP_RIGHT_TARGET"
    
    # Stack/Occlusion Resolved
    GRASP_BOTTOM = "GRASP_BOTTOM"         # Target was at bottom
    GRASP_VERTICAL = "GRASP_VERTICAL"     # Partial occlusion strategy
    
    # Default
    GRASP_STANDARD = "GRASP_STANDARD"


class InstructionRegistry:
    """
    Central repository for robot instructions.
    - 'robot': The prompt sent to the motion planner/controller.
    - 'ui': The message displayed to the human operator.
    """
    INSTRUCTIONS = {
        # --- Negative Zone (Stop) ---
        ActionCodes.EMERGENCY_STOP: {
            "robot": None, 
            "ui": "⚠️ EMERGENCY STOP: Human detected!"
        },
        ActionCodes.ABORT_TARGET_MISSING: {
            "robot": None, 
            "ui": "�?Error: Target not found in view."
        },
        
        # --- Boundary Zone (Intervention/Pre-process) ---
        ActionCodes.REQ_USER_SELECT: {
            "robot": None, 
            "ui": "�?Ambiguity detected. Select target: Left or Right?"
        },
        ActionCodes.ACT_REMOVE_TOP: {
            "robot": "Pick up the obstacle", 
            "ui": "⚠️ Stack detected: Removing top object..."
        },
        ActionCodes.ACT_CLEAR_PATH: {
            "robot": "Move the obstacle", 
            "ui": "⚠️ Path blocked: Clearing obstacle..."
        },

        # --- Positive Zone (Execute) ---
        ActionCodes.GRASP_LEFT_TARGET: {
            "robot": "Grasp the blue cube on the left", 
            "ui": "�?Executing: Grasping left target."
        },
        ActionCodes.GRASP_RIGHT_TARGET: {
            "robot": "Grasp the blue cube on the right", 
            "ui": "�?Executing: Grasping right target."
        },
        ActionCodes.GRASP_BOTTOM: {
            "robot": "Grasp the blue cube", 
            "ui": "�?Stack cleared: Grasping target."
        },
        ActionCodes.GRASP_VERTICAL: {
            "robot": "Grasp the blue cube", 
            "ui": "👀 Partial occlusion: Switching to vertical grasp strategy."
        },
        ActionCodes.GRASP_STANDARD: {
            "robot": "Grasp the blue cube", 
            "ui": "�?Path clear: Executing standard grasp."
        },
    }

    # 这里的描述是专门给大模型（Brain）看�?"菜单"
    # 我们�?Decision Zone 直接写进描述里，引导模型做分�?# 这里的描述直接决定了大模型的决策边界
    BRAIN_DESC = {
        # Negative (一票否�?
        ActionCodes.EMERGENCY_STOP:       "CRITICAL SAFETY: Use this ONLY if a human, hand, or person is detected in the workspace.",
        ActionCodes.ABORT_TARGET_MISSING: "FEASIBILITY ERROR: Use this if the target object is NOT found in either Global or Wrist images.",
        
        # Boundary (歧义与物理约�?- 核心改进�?
        ActionCodes.REQ_USER_SELECT:      "LOGIC CONFLICT: Use this if MULTIPLE identical targets are present (e.g., 2 blue cubes) but the instruction does NOT specify 'left', 'right', or 'top'. DO NOT pick one randomly.",
        ActionCodes.ACT_REMOVE_TOP:       "PHYSICAL CONSTRAINT: Target is clearly visible but UNDER another object. Must remove the top one first.",
        ActionCodes.ACT_CLEAR_PATH:       "PHYSICAL CONSTRAINT: A large obstacle is blocking the direct reach path. Path is inaccessible.",
        
        # Positive (执行策略)
        ActionCodes.GRASP_VERTICAL:       "STRATEGY: Use this if the target is in a crowded area but can be reached safely from directly above.",
        ActionCodes.GRASP_LEFT_TARGET:    "SPECIFIC: Use this when the user specifically asks for the LEFT target, or to resolve an earlier ambiguity.",
        ActionCodes.GRASP_RIGHT_TARGET:   "SPECIFIC: Use this when the user specifically asks for the RIGHT target, or to resolve an earlier ambiguity.",
        ActionCodes.GRASP_BOTTOM:         "STRATEGY: Standard grasp for the target object at the bottom after the top is cleared.",
        
        # Default
        ActionCodes.GRASP_STANDARD:       "DEFAULT: Use this ONLY when there is exactly ONE target, NO obstacles, and NO humans. Path must be 100% clear.",
    }

    

    @staticmethod
    def get_instruction(action_code, **kwargs):
        """获取给机器人的指令和UI信息"""
        template = InstructionRegistry.INSTRUCTIONS.get(action_code, {})
        res = {"robot": None, "ui": "Unknown Action"}
        
        # 填充 robot 指令中的 {target} 参数
        if template.get("robot"):
            try: 
                res["robot"] = template["robot"].format(**kwargs)
            except: 
                res["robot"] = template["robot"]
                
        if template.get("ui"): 
            res["ui"] = template["ui"]
            
        return res

    @staticmethod
    def get_brain_menu():
        """生成结构化的指令协议，强制模型进行逻辑优先级判�?""
        menu = "### ROBOTIC LOGIC PROTOCOL (FOLLOW PRIORITY) ###\n"
        
        # 按照优先级强制排序，并在描述中加入“判定准则�?        categories = [
            ("PRIORITY 0: SAFETY & EMERGENCY", [ActionCodes.EMERGENCY_STOP]),
            ("PRIORITY 1: EXISTENCE CHECK", [ActionCodes.ABORT_TARGET_MISSING]),
            ("PRIORITY 2: AMBIGUITY & CONSTRAINTS", [ActionCodes.REQ_USER_SELECT, ActionCodes.ACT_REMOVE_TOP, ActionCodes.ACT_CLEAR_PATH]),
            ("PRIORITY 3: EXECUTION STRATEGY", [ActionCodes.GRASP_VERTICAL, ActionCodes.GRASP_LEFT_TARGET, ActionCodes.GRASP_RIGHT_TARGET, ActionCodes.GRASP_STANDARD])
        ]
        
        for cat_name, codes in categories:
            menu += f"\n{cat_name}:\n"
            for code in codes:
                desc = InstructionRegistry.BRAIN_DESC.get(code, "No description.")
                menu += f"  - [{code}]: {desc}\n"
        
        menu += "\n**DECISION RULE**: If multiple objects match the description but NO spatial keyword (left/right) is provided, you MUST return REQ_USER_SELECT."
        return menu


