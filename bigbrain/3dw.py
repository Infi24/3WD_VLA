import sys, re, os, torch
import numpy as np
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

# ==============================================================================
# 1. 核心数学模块：Alpha Calculation (Only)
# ==============================================================================
def calculate_alpha_only(preference_str, actions=None):
    """
    解析模型输出的 Direct/Ask/Replan/Reject 两两比较，只计算 Alpha 值。
    Alpha > 0.85 -> Positive
    """
    if actions is None:
        actions = {'direct': 0, 'ask': 1, 'replan': 2, 'reject': 3}
    
    n = len(actions)
    w = np.zeros((n, n))
    
    # 正则提取：忽略大小写，容忍空格
    pairs = re.findall(r'\b(direct|ask|replan|reject)\s*>\s*(direct|ask|replan|reject)\b', preference_str.lower())
    
    if not pairs:
        return None

    for winner, loser in pairs:
        if winner in actions and loser in actions:
            # 限制权重上限，防止重复生成导致的数值倾斜
            current_val = w[actions[winner]][actions[loser]]
            w[actions[winner]][actions[loser]] = min(current_val + 1, 1.5) 
            
    # MM 算法求解 Bradley-Terry 模型 (迭代求胜率分布 Pi)
    W_vector = np.sum(w, axis=1)
    nij_matrix = w + w.T
    pi = np.ones(n) 
    
    # 20次迭代通常足够收敛
    for _ in range(20):
        pi_old = pi.copy()
        for i in range(n):
            denom = 0
            for j in range(n):
                if i != j and nij_matrix[i, j] > 0:
                    denom += nij_matrix[i, j] / (pi_old[i] + pi_old[j])
            if denom > 0:
                pi[i] = W_vector[i] / denom
        
        pi_sum = np.sum(pi)
        if pi_sum > 0: pi /= pi_sum
        else: pi = np.ones(n) / n 

        if np.linalg.norm(pi - pi_old, ord=1) < 1e-6:
            break
            
    pi_map = {name: pi[idx] for name, idx in actions.items()}
    
    # 计算 Alpha
    # Alpha = P_rel * P_aff
    # P_rel: 语义轴 (Direct vs Ask)
    # P_aff: 物理轴 (Direct vs Replan/Reject)
    epsilon = 1e-9
    pi_d, pi_a = pi_map['direct'], pi_map['ask']
    pi_r, pi_rej = pi_map['replan'], pi_map['reject']
    
    p_rel = pi_d / (pi_d + pi_a + epsilon)
    p_aff = pi_d / (pi_d + pi_r + pi_rej + epsilon)
    
    alpha = p_rel * p_aff
    return alpha

# ==============================================================================
# 2. 前置门控解析 (Negative Domain Check)
# ==============================================================================
def parse_precheck(response_text):
    """
    解析 Step 1 的前置回答。
    逻辑：必须满足 (Target Visible=Yes) AND (Human Hand=No) 才能进入计算。
    """
    response_lower = response_text.lower()
    
    # 提取 Q1 (Object Visible)
    obj_match = re.search(r'visible\??\s*[:\-]?\s*(yes|no)', response_lower)
    # 提取 Q2 (Human Hand)
    hand_match = re.search(r'hand\??\s*[:\-]?\s*(yes|no)', response_lower)
    
    obj_status = obj_match.group(1) if obj_match else "unknown"
    hand_status = hand_match.group(1) if hand_match else "unknown"
    
    # 严格门控
    if obj_status == "yes" and hand_status == "no":
        return True, f"PASS (Target:{obj_status}, Hand:{hand_status})"
    else:
        return False, f"FAIL (Target:{obj_status}, Hand:{hand_status})"

# ==============================================================================
# 3. 推理主流程
# ==============================================================================
def run_inference(img_path, model_path, adapter_path):
    print(f"🚀 [Init] Loading Model: {model_path}")
    print(f"🔗 [Init] Loading Adapter: {adapter_path}")
    
    # 加载模型 (BF16 加速)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    
    processor = AutoProcessor.from_pretrained(model_path)

    # Prompt Engineering: 分步引导
    user_text = (
        "Analyze the image for robotic grasping task.\n"
        "Step 1: Answer the pre-check questions with Yes or No.\n"
        "   - Q1: Is the blue cube target visible?\n"
        "   - Q2: Is there a human hand intervening or very close to the target?\n"
        "Step 2: Output the pairwise priority among: direct, ask, replan, reject.\n"
        "   Format: 'direct>reject ask>direct ...'\n"
        "\nResponse:"
    )

    messages = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": user_text}]}]
    
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

    print(f"🧠 [Infer] Processing image: {os.path.basename(img_path)}")
    
    with torch.no_grad():
        # 强制确定性生成，不采样
        generated_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    print(f"\n📝 [Output] Model Response:\n{response}\n" + "-" * 50)
    
    # --- 阶段 1: 负域检测 ---
    is_safe, check_msg = parse_precheck(response)
    if not is_safe:
        print(f"🛑 [Decision] NEGATIVE (Stop)")
        print(f"   Reason: {check_msg}")
        return "NEGATIVE"

    # --- 阶段 2: 计算 Alpha ---
    print(f"✅ [Pre-check] Passed. Calculating Alpha...")
    alpha = calculate_alpha_only(response)
    
    if alpha is None:
        print("⚠️ [Warning] No valid pairwise comparison found.")
        return "BOUNDARY"

    # --- 阶段 3: 阈值判定 ---
    decision = "POSITIVE" if alpha > 0.85 else "BOUNDARY"
    
    print(f"📊 [Score] Alpha: {alpha:.4f}")
    print(f"🎯 [Decision] {decision}")
    
    return decision

if __name__ == "__main__":
    # 默认配置路径
    BASE_MODEL = "/root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct"
    # 这里指向你的 Output_Dir
    ADAPTER = "/root/autodl-tmp/bigbrain/saves/qwen3_8b_grasp_final" 
    # 测试图片
    IMAGE = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/test/23.jpg"
    
    run_inference(IMAGE, BASE_MODEL, ADAPTER)