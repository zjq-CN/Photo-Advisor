"""策略/引导文本构建。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
from typing import Any, Callable, Dict, List, Optional, Tuple
from app.contracts import DEFAULT_USER_INTENT, OUTPUT_SIZE_PRESETS, intent_option_label



def infer_automatic_goal(ai_result: Dict[str, str]) -> str:
    evidence = " ".join([
        ai_result.get("subject_position_analysis", ""),
        ai_result.get("camera_angle_analysis", ""),
        ai_result.get("shot_size_analysis", ""),
        ai_result.get("composition_analysis", ""),
        ai_result.get("light_source_inference", ""),
        ai_result.get("suggested_adjustment", ""),
    ]).lower()
    keywords = {
        "clarity": ["模糊", "对焦", "噪点", "噪声", "不清晰", "抖动"],
        "lighting": ["过暗", "过曝", "逆光", "曝光", "阴影", "高光", "肤色", "白平衡"],
        "background": ["背景杂乱", "背景干扰", "杂物", "遮挡", "边缘干扰"],
        "subject": ["人物太小", "主体不突出", "人物不突出", "主体占比"],
        "composition": ["构图", "留白", "偏左", "偏右", "过高", "过低", "倾斜", "水平线", "垂直线"],
    }
    scores = {
        goal: sum(evidence.count(word) for word in words)
        for goal, words in keywords.items()
    }
    best_goal = max(scores, key=scores.get)
    return best_goal if scores[best_goal] > 0 else "composition"


GOAL_UPDATES: Dict[str, Dict[str, str]] = {
        "composition": {
            "composition_goal": "修正主体位置、头顶留白、画面边缘与水平垂直关系",
            "composition_methods": "优先裁切和轻微校正；仅在原图已有线条时利用三分法或引导线",
        },
        "subject": {
            "composition_goal": "在不换景的前提下提高人物视觉权重",
            "subject_position_target": "人物靠近视觉中心或合适三分线",
            "subject_ratio_target": "通过自然裁切适度提高人物占比",
            "depth_strategy": "人物边缘清晰，背景只做轻微弱化",
            "composition_methods": "主体占比、边缘减扰、人物与背景明度层级",
        },
        "lighting": {
            "composition_goal": "保持构图，优先改善曝光、面部明暗与真实肤色",
            "light_strategy": "恢复高光和暗部细节，均衡面部曝光，不改变原光源方向",
            "color_strategy": "校正白平衡和偏色，肤色自然，不增白或涂抹皮肤质感",
            "composition_methods": "曝光层次、高光保护、暗部细节与白平衡",
        },
        "background": {
            "composition_goal": "通过裁切、边缘整理和轻微背景柔化减少干扰",
            "negative_space_target": "减少无效留白和边缘杂乱，不删除关键场景元素",
            "depth_strategy": "人物清晰，背景轻微柔化但仍可读",
            "composition_methods": "边缘裁切、背景明度压低和克制的景深分离",
        },
        "clarity": {
            "composition_goal": "保持原构图，提升人物和关键细节的清晰度",
            "depth_strategy": "主体细节清楚，避免锐化光晕、条纹和假纹理",
            "composition_methods": "轻微降噪、局部细节恢复和克制锐化",
        },
    }


SHOT_UPDATES: Dict[str, Dict[str, str]] = {
        "keep": {},
        "headshot": {
            "shot_size_target": "头肩特写",
            "subject_ratio_target": "人物占画面约60%到78%",
            "subject_position_target": "双眼位于上三分区域，头顶留白自然",
            "lens_feel": "自然头像透视，不改变脸型",
        },
        "half_body": {
            "shot_size_target": "半身人像",
            "subject_ratio_target": "人物占画面约45%到65%",
            "subject_position_target": "头肩与上半身完整，关节位置不被生硬裁断",
        },
        "full_body": {
            "shot_size_target": "全身人像",
            "subject_ratio_target": "人物占画面约35%到55%",
            "subject_position_target": "头部到脚部完整保留，四肢比例真实",
        },
        "environmental": {
            "shot_size_target": "环境人像",
            "subject_ratio_target": "人物占画面约20%到40%",
            "subject_position_target": "人物与原有环境建立清晰关系",
            "negative_space_target": "保留可读的环境和叙事留白",
            "lens_feel": "自然环境人像透视，不伪造广角拉伸",
        },
    }


MOOD_UPDATES: Dict[str, str] = {
        "keep": "除完成主目标所需的轻微纠偏外，保持原图色温、饱和度和对比关系",
        "natural": "自然中性、通透但不过曝，肤色真实",
        "warm": "在原图基础上轻微增暖，肤色不偏黄不偏红",
        "fresh": "明度轻盈、色彩干净，保留高光细节",
        "cool": "轻微冷调与克制饱和度，人物肤色仍保持真实",
        "cinematic": "使用原图已有明暗形成克制的电影对比，不新增光源或色彩特效",
    }


STRENGTH_UPDATES: Dict[str, str] = {
        "conservative": (
            "只做一档轻微调整；优先保留原图构图、姿态、表情和色彩；"
            "如果优化会导致换人或换景，则放弃该调整。"
        ),
        "standard": (
            "执行可感知但自然的优化；可适度裁切、校正和调整明暗，"
            "不重建人物、姿态或场景。"
        ),
        "strong": (
            "允许更明显的裁切、主体占比、曝光和层次优化；"
            "仍必须保持同一人物、原有表情、真实身体比例和原场景关键元素。"
        ),
    }


def _build_base_strategy(
    user_intent: Dict[str, str],
    requested_goal: str,
    effective_goal: str,
) -> Dict[str, str]:
    """基础策略骨架（标签来自意图选项，值与原实现逐字相同）。"""
    return {
        "requested_goal": requested_goal,
        "effective_goal": effective_goal,
        "optimization_goal_label": intent_option_label("optimization_goal", requested_goal),
        "effective_goal_label": intent_option_label("optimization_goal", effective_goal),
        "shot_type_label": intent_option_label("shot_type", user_intent["shot_type"]),
        "mood_style_label": intent_option_label("mood_style", user_intent["mood_style"]),
        "output_ratio_label": intent_option_label("output_ratio", user_intent["output_ratio"]),
        "edit_strength_label": intent_option_label("edit_strength", user_intent["edit_strength"]),
        "composition_goal": "保持主体清晰、画面平衡和真实场景",
        "camera_height": "保持原机位",
        "camera_angle": "保持原角度，仅做轻微透视校正",
        "shot_size_target": "保持原景别",
        "lens_feel": "保持原图自然透视",
        "subject_position_target": "保持人物可识别性，调整到稳定视觉位置",
        "subject_ratio_target": "尽量保持原图人物占比",
        "negative_space_target": "留白均衡且有明确作用",
        "light_strategy": "仅调整原图已有光线的曝光与明暗层次，不新增光源",
        "depth_strategy": "保持真实纵深，不伪造镜头光学效果",
        "pose_strategy": "保留原有姿态与身体比例，只做必要的轻微调整",
        "expression_hint": "保留原图表情类型和嘴部闭合状态，不强行改表情",
        "eye_focus_strategy": "保留原视线，只优化眼部清晰度和自然高光",
        "composition_methods": "裁切、留白和水平垂直线校正",
        "color_strategy": "保持真实肤色与自然白平衡",
        "geometry_strategy": "保持人脸、身体和环境的真实几何比例",
    }


def build_strategy_plan(ai_result: Dict[str, str], user_intent: Dict[str, str]) -> Dict[str, str]:
    requested_goal = user_intent["optimization_goal"]
    effective_goal = infer_automatic_goal(ai_result) if requested_goal == "auto" else requested_goal

    strategy = _build_base_strategy(user_intent, requested_goal, effective_goal)
    strategy.update(GOAL_UPDATES[effective_goal])
    strategy.update(SHOT_UPDATES[user_intent["shot_type"]])

    mood_strategy = MOOD_UPDATES[user_intent["mood_style"]]
    if effective_goal == "lighting":
        # 光线主目标需要保留白平衡/肤色修复要求；氛围只决定调色方向，
        # 不能像旧版那样把光线策略整体覆盖掉。
        strategy["color_strategy"] = f"{strategy['color_strategy']}；氛围控制：{mood_strategy}"
    else:
        strategy["color_strategy"] = mood_strategy

    strategy["strength_rule"] = STRENGTH_UPDATES[user_intent["edit_strength"]]

    light_text = ai_result.get("light_source_inference", "")
    if "不明显" in light_text or "无法" in light_text:
        strategy["light_strategy"] += " 原图光源方向不可靠，不制造新的窗光、轮廓光或逆光。"

    return strategy


def build_guidance_text(
    ai_result: Dict[str, str],
    strategy_plan: Dict[str, str],
    user_intent: Dict[str, str]
) -> Dict[str, str]:
    _ = user_intent
    auto_note = ""
    if strategy_plan["requested_goal"] == "auto":
        auto_note = f"自动识别主要目标为“{strategy_plan['effective_goal_label']}”。"
    return {
        "scene_summary": ai_result.get("scene_summary", "已完成基础诊断。"),
        "subject_gender": ai_result.get("subject_gender", "未返回该项结果。"),
        "subject_position_analysis": ai_result.get("subject_position_analysis", "未返回该项结果。"),
        "camera_angle_analysis": ai_result.get("camera_angle_analysis", "未返回该项结果。"),
        "shot_size_analysis": ai_result.get("shot_size_analysis", "未返回该项结果。"),
        "composition_analysis": ai_result.get("composition_analysis", "未返回该项结果。"),
        "light_source_inference": ai_result.get("light_source_inference", "未返回该项结果。"),
        "recommended_shooting_position": (
            f"{auto_note}建议按“{strategy_plan['effective_goal_label']}”优先调整；"
            f"{strategy_plan['camera_height']}，{strategy_plan['camera_angle']}，{strategy_plan['subject_position_target']}。"
        ),
        "suggested_adjustment": (
            f"构图：{strategy_plan['composition_methods']}；"
            f"景别：{strategy_plan['shot_size_target']}；"
            f"光线：{strategy_plan['light_strategy']}；"
            f"色调：{strategy_plan['color_strategy']}；"
            f"强度：{strategy_plan['strength_rule']}"
        ),
        "expression_hint": strategy_plan["expression_hint"],
        "ideal_image_prompt": ai_result.get("ideal_image_prompt", "未返回该项结果。"),
    }


def build_generation_settings(user_intent: Dict[str, str]) -> Dict[str, Any]:
    ratio_key = user_intent.get("output_ratio", DEFAULT_USER_INTENT["output_ratio"])
    preset = OUTPUT_SIZE_PRESETS.get(ratio_key, OUTPUT_SIZE_PRESETS[DEFAULT_USER_INTENT["output_ratio"]])
    return {
        "output_ratio": ratio_key,
        "output_ratio_label": intent_option_label("output_ratio", ratio_key),
        "dashscope_size": preset["dashscope_size"],
        "doubao_size": preset["doubao_size"],
        "expected_ratio": preset["expected_ratio"],
        "edit_strength": user_intent.get("edit_strength", DEFAULT_USER_INTENT["edit_strength"]),
    }
