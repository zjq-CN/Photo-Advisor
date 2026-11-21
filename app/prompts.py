"""全部提示词构建（纯字符串，内容冻结）。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
from typing import Any, Callable, Dict, List, Optional, Tuple



# =========================
# 第一次分析提示词
# =========================
def build_analysis_prompt(geometry_block: str = "") -> str:
    prompt = """
你是专业的人像摄影指导，用户用手机拍摄。分析照片后，只输出一个 JSON 对象（不要 markdown 代码块，不要任何解释文字），key 必须如下：

{
  "scene_summary": "画面简短概括，1-2句；多人时点明主体人物",
  "main_issue": "这张照片最需要解决的一个问题，一句话；没有明显问题时写\"无明显问题\"",
  "subject_gender": "仅填\"男性\"或\"女性\"，按照片真实外观；多人时填最主体的人物（通常是面部最大或最清晰者）",
  "subject_position_analysis": "主体人物位置（偏左/右/居中、高低、留白是否合理）",
  "camera_angle_analysis": "拍摄角度（平拍/俯拍/仰拍、正面/侧面）及是否合适，并说明该角度对人物观感的影响",
  "shot_size_analysis": "景别（头像/半身/全身/环境人像）、人物占比、裁切是否合理",
  "composition_analysis": "构图诊断：头顶留白、人物偏移、前景遮挡、水平垂直线、背景干扰，以及画质问题（模糊/糊脸/过暗过曝）",
  "light_source_inference": "主光源方向及依据；若明暗关系不明显、无法可靠判断，写\"光源方向不明显\"，不要强行推测",
  "recommended_shooting_position": "拍摄者下一步怎么移动，用口语化动作描述（往哪走、蹲多低、手机举多高），不用影棚术语",
  "suggested_adjustment": "人物姿态与拍摄者分别怎么调整，动作要当场可做",
  "ideal_image_prompt": "按建议调整完成后的理想画面描述（见下方要求）"
}

字段写法：每个诊断字段按\"现象 → 背后的摄影原则 → 怎么改\"组织；做得好的地方要点明它符合了什么原则。照片有多人时只分析画面主体，不逐人展开。

ideal_image_prompt 要求：
1. 描述\"调整后\"的理想画面，不是当前照片，不是建议或教学说明；
2. 重点描述人物理想位置、理想角度、整体观感，不详细描述环境（编辑模型会参考原图）；
3. 仅在能明确判断光源方向时描述光线，否则不提光源；
4. 真实照片效果，非卡通/插画，无箭头、图标或标注文字，人物性别与 subject_gender 一致。

整体：输出中文；建议要具体到当场能做。
""".strip()
    if geometry_block:
        return prompt + "\n\n" + geometry_block
    return prompt


def build_targeted_ideal_prompt(
    ai_result: Dict[str, str],
    strategy_plan: Dict[str, str],
    user_intent: Dict[str, str]
) -> str:
    auto_note = ""
    if strategy_plan["requested_goal"] == "auto":
        auto_note = f"系统根据诊断将主要问题确定为“{strategy_plan['effective_goal_label']}”。"
    return (
        f"{auto_note}结果是一张真实摄影照片，优先完成“{strategy_plan['effective_goal_label']}”。"
        f"景别为{strategy_plan['shot_size_target']}，人物位置为{strategy_plan['subject_position_target']}，"
        f"构图目标为{strategy_plan['composition_goal']}。"
        f"光线处理为{strategy_plan['light_strategy']}，色调为{strategy_plan['color_strategy']}。"
        f"输出画幅为{strategy_plan['output_ratio_label']}，优化强度为{strategy_plan['edit_strength_label']}。"
        "始终保持同一人物、原有表情类型、发型、服装、配饰、身体比例和原场景关键元素。"
    )


def build_edit_prompt(
    ai_result: Dict[str, str],
    strategy_plan: Dict[str, str],
    user_intent: Dict[str, str],
    targeted_ideal_prompt: str
) -> str:
    return f"""
请对输入原图做一次真实、可执行、克制的摄影优化，生成“调整后的理想拍摄结果图”。

用户明确选择（必须实际体现）：
- 优先改善：{strategy_plan.get("optimization_goal_label", "")}
- 实际执行目标：{strategy_plan.get("effective_goal_label", "")}
- 目标景别：{strategy_plan.get("shot_type_label", "")}
- 画面氛围：{strategy_plan.get("mood_style_label", "")}
- 输出画幅：{strategy_plan.get("output_ratio_label", "")}
- 优化强度：{strategy_plan.get("edit_strength_label", "")}

目标结果：
{targeted_ideal_prompt}

专业执行策略：
- 构图：{strategy_plan.get("composition_goal", "")}
- 景别：{strategy_plan.get("shot_size_target", "")}
- 人物位置：{strategy_plan.get("subject_position_target", "")}
- 人物占比：{strategy_plan.get("subject_ratio_target", "")}
- 留白：{strategy_plan.get("negative_space_target", "")}
- 构图方法：{strategy_plan.get("composition_methods", "")}
- 光线：{strategy_plan.get("light_strategy", "")}
- 色调：{strategy_plan.get("color_strategy", "")}
- 清晰度与空间：{strategy_plan.get("depth_strategy", "")}
- 几何比例：{strategy_plan.get("geometry_strategy", "")}
- 人物姿态：{strategy_plan.get("pose_strategy", "")}
- 表情与视线：{strategy_plan.get("expression_hint", "")}；{strategy_plan.get("eye_focus_strategy", "")}
- 强度规则：{strategy_plan.get("strength_rule", "")}

原图诊断依据：
- 人物位置：{ai_result.get("subject_position_analysis", "")}
- 当前角度：{ai_result.get("camera_angle_analysis", "")}
- 构图问题：{ai_result.get("composition_analysis", "")}
- 光线判断：{ai_result.get("light_source_inference", "")}
- 诊断阶段的理想描述（仅作参考，不得覆盖用户选择）：{ai_result.get("ideal_image_prompt", "")}

最高优先级约束：
1. 必须是同一人；不换脸、不换人，不改年龄、性别、脸型、体型和五官比例。
2. 保留原有表情类型、嘴部闭合状态、发型、服装、眼镜和配饰。
3. 保留原场景的背景、道具、空间关系和关键元素；不新增、删除、替换主要内容。
4. 改变画幅时优先安全裁切和重排画面权重，不补画原图外不存在的景物。
5. 只能利用原图已有光线；不新增窗光、阳光、轮廓光、补光灯、霓虹灯或光斑。
6. 保持真实皮肤纹理和毛孔；不增白、不整容、不过度磨皮、不使用网红滤镜。
7. 不要卡通、插画、动漫、3D 渲染、海报化或虚假电影特效。
8. 图中不添加任何文字、箭头、标签、气泡框、对话框或教学标注。
9. 如果某项优化会导致换人、换景、肢体变形或虚构元素，必须放弃该项调整，优先保持原图真实性。
""".strip()

QWEN_IMAGE_NEGATIVE_PROMPT = (
    "换脸,换人,陌生人,性别错误,年龄改变,脸型改变,体型改变,发型改变,服装改变,配饰改变,"
    "改变原表情,嘴部变形,眼睛变形,五官变形,多余牙齿,肢体变形,卡通风格,插画风格,动漫风格,二次元,3d渲染,"
    "气泡对话框,对白框,教学标注,箭头,标签,海报标题,大段文字,遮挡主体的文字,整容脸,塑料皮肤,过度磨皮,"
    "条纹噪点,网纹,摩尔纹,扫描线,网格噪声,纹理污染,呆滞眼神,空洞眼神,失焦眼神,斗鸡眼"
)
