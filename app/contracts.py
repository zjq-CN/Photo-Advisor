"""字段名常量、OPTIMIZATION_SCHEMA、意图归一化。

阶段③ 从本文件拆出解析类纯函数 → app/parsing.py（P0 单文件 ≤400 行）。
常量与意图归一化逻辑逐字节未改。
"""

from typing import Any, Dict, List

from app.config import ZHIPU_REASONING_EFFORT


# 视觉诊断模式（前端拍照页可选，/api/diagnose 的 mode 字段）：模式 = 模型链顺序 + 每模型思考参数覆盖
# - standard（默认）：glm-5.3-flash 主力（质量优先，压测 p50 26.3s / p95 33.9s），
#   deepseek 思考档作质量救火（用户拍板：兜底模型保留思考）；
# - fast：qwen3.5-flash 主力（关思考 ~5-10s），deepseek 关思考守速度承诺，glm 退居慢速救火。
# 链内同一供应商（阿里百炼）的 qwen 模型错开，避免百炼整体限流时连续失败。
# 2026-08-31 晚：修复 qwen extra_body 死参数 + 关闭 deepseek 默认高档思考后，qwen 实测 ~5-10s、deepseek ~4-5s（均零思考）。
# API Key 配置保留在 _config/api_keys.json，换模型只需调整本表，无需改 Key 文件。
VISION_MODE_PRESETS = {
    "standard": {
        "chain": [
            "glm-5.3-flash",
            "deepseek-flash",
            "qwen3-vl-plus",
            "hy-vision-2.0-instruct",
            "qwen3.8-flash",
        ],
        "params": {
            "glm-5.3-flash": {"reasoning_effort": ZHIPU_REASONING_EFFORT},
            "deepseek-flash": {"thinking": {"type": "enabled"}},
            "qwen3.8-flash": {"enable_thinking": False},
        },
    },
    "fast": {
        "chain": [
            "qwen3.8-flash",
            "deepseek-flash",
            "glm-5.3-flash",
            "qwen3-vl-plus",
            "hy-vision-2.0-instruct",
        ],
        "params": {
            "deepseek-flash": {"thinking": {"type": "disabled"}},
            "qwen3.8-flash": {"enable_thinking": False},
            "glm-5.3-flash": {"reasoning_effort": ZHIPU_REASONING_EFFORT},
        },
    },
}
VISION_MODELS = VISION_MODE_PRESETS["standard"]["chain"]  # 兼容旧引用（run_exam.py、启动打印等）

# 图像编辑模型：推荐顺序（主 -> 备）
# 经 2026-08 生图模型考试筛选，保留三个互为补充的模型：
# 通义万相最快、豆包 Seedream 人像真实感强、qwen-image-3.0-pro 生图编辑二合一。
# qwen-image-edit-max 与即梦 4.0 经对比不再保留（代码 runner 仍保留，便于日后切换）。
MODEL_WAN_27 = "wan2.7-image"
MODEL_QWEN_IMAGE_20 = "qwen-image-2.0"
MODEL_SEEDREAM_50 = "doubao-seedream-5.0"
MODEL_QWEN_EDIT_MAX = "qwen-image-edit-max"

IMAGE_EDIT_MODELS = [
    MODEL_WAN_27,
    MODEL_SEEDREAM_50,
    MODEL_QWEN_IMAGE_20,
]
MODEL_COMPARE_ENABLED_DEFAULT = True
MODEL_COMPARE_LIST = list(IMAGE_EDIT_MODELS)

# 优化选项的唯一数据源。前端从这里渲染，后端也用同一份数据验证，
# 避免再出现“界面有选项，后端却没有实现”的版本错位。
OPTIMIZATION_SCHEMA: List[Dict[str, Any]] = [
    {
        "key": "optimization_goal",
        "label": "优先改善",
        "help": "选一个最想解决的问题；不确定时交给系统自动判断。",
        "options": [
            {"value": "auto", "label": "按诊断自动优化", "hint": "优先修正基础诊断中最明显的问题"},
            {"value": "composition", "label": "构图与裁切", "hint": "调整主体位置、留白、水平线和画面边缘"},
            {"value": "subject", "label": "人物更突出", "hint": "提高人物权重，弱化背景干扰"},
            {"value": "lighting", "label": "光线与肤色", "hint": "优化曝光、明暗层次、白平衡与真实肤色"},
            {"value": "background", "label": "背景减干扰", "hint": "通过裁切和轻微柔化减少杂乱，不删除关键元素"},
            {"value": "clarity", "label": "清晰度与降噪", "hint": "优化对焦感、细节和暗部噪点，避免过度锐化"},
        ],
    },
    {
        "key": "shot_type",
        "label": "目标景别",
        "help": "用直观的景别代替 35mm/50mm/85mm 等难以稳定模拟的镜头术语。",
        "options": [
            {"value": "keep", "label": "保持原景别", "hint": "尽量不改变原图人物占比"},
            {"value": "headshot", "label": "头像特写", "hint": "头肩为主，面部识别度优先"},
            {"value": "half_body", "label": "半身人像", "hint": "头肩与上半身姿态兼顾"},
            {"value": "full_body", "label": "全身人像", "hint": "保留完整姿态和自然身体比例"},
            {"value": "environmental", "label": "环境人像", "hint": "保留更多环境信息与空间层次"},
        ],
    },
    {
        "key": "mood_style",
        "label": "画面氛围",
        "help": "只调整色调和对比，不强行改变人物表情。",
        "options": [
            {"value": "keep", "label": "保持原色", "hint": "保留原图色温与氛围"},
            {"value": "natural", "label": "自然通透", "hint": "中性真实，明暗清晰"},
            {"value": "warm", "label": "温暖柔和", "hint": "轻微暖色，降低生硬感"},
            {"value": "fresh", "label": "清爽明亮", "hint": "色彩干净轻盈，不过曝"},
            {"value": "cool", "label": "冷调克制", "hint": "轻微冷调，保持真实肤色"},
            {"value": "cinematic", "label": "电影质感", "hint": "克制的对比和层次，不新增戏剧光源"},
        ],
    },
    {
        "key": "output_ratio",
        "label": "输出画幅",
        "help": "4:3 与项目的 320×240 屏幕直接匹配，可避免二次旋转和大幅裁切。",
        "options": [
            {"value": "source", "label": "跟随原图", "hint": "尽量保持原图横竖比例"},
            {"value": "screen_4_3", "label": "4:3 设备屏幕", "hint": "直接适配 320×240 横屏"},
            {"value": "portrait_3_4", "label": "3:4 竖版人像", "hint": "适合竖屏人像展示"},
            {"value": "square_1_1", "label": "1:1 方形头像", "hint": "适合头像和方形预览"},
            {"value": "landscape_16_9", "label": "16:9 横向环境", "hint": "适合展示环境和叙事留白"},
        ],
    },
    {
        "key": "edit_strength",
        "label": "优化强度",
        "help": "控制裁切、透视、色彩和背景弱化幅度。",
        "options": [
            {"value": "conservative", "label": "保守", "hint": "最大限度保留原图，只修正明显问题"},
            {"value": "standard", "label": "标准", "hint": "可感知但自然的优化"},
            {"value": "strong", "label": "明显", "hint": "更明显的裁切与层次优化，仍不换人换景"},
        ],
    },
]

DEFAULT_USER_INTENT: Dict[str, str] = {
    "optimization_goal": "auto",
    "shot_type": "keep",
    "mood_style": "keep",
    "output_ratio": "screen_4_3",
    "edit_strength": "standard",
}

OPTIMIZATION_OPTION_MAP: Dict[str, Dict[str, Dict[str, str]]] = {
    group["key"]: {option["value"]: option for option in group["options"]}
    for group in OPTIMIZATION_SCHEMA
}


# =========================
# 第二次图像编辑提示词
# =========================
def normalize_user_intent(raw_intent: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(raw_intent, dict):
        raw_intent = {}

    legacy_goal_map = {
        "更自然耐看": "auto",
        "人物更突出": "subject",
        "背景虚化分离": "background",
        "背景更壮观": "composition",
        "引导线纵深": "composition",
        "框架构图": "composition",
        "前景层次": "composition",
        "画面更干净": "background",
        "对称稳定": "composition",
        "低角度张力": "composition",
        "显腿更长": "composition",
        "逆光轮廓": "lighting",
        "更像电影镜头": "composition",
    }
    legacy_mood_map = {
        "温暖": "warm",
        "清爽": "fresh",
        "冷感": "cool",
        "高反差戏剧": "cinematic",
        "平静": "natural",
    }
    legacy_shot_map = {
        "日常头像": "headshot",
        "职业头像": "headshot",
        "社交头像": "headshot",
        "形象照": "half_body",
        "85mm半身人像": "half_body",
        "旅行打卡": "environmental",
        "街拍人像": "environmental",
        "夜景人像": "environmental",
        "校园记录": "environmental",
        "毕业纪念": "environmental",
        "35mm环境人像": "environmental",
    }

    migrated: Dict[str, Any] = dict(raw_intent)
    if not migrated.get("optimization_goal"):
        migrated["optimization_goal"] = legacy_goal_map.get(
            str(migrated.get("visual_goal", "")).strip(),
            DEFAULT_USER_INTENT["optimization_goal"],
        )
    if not migrated.get("mood_style"):
        migrated["mood_style"] = legacy_mood_map.get(
            str(migrated.get("emotion_style", "")).strip(),
            DEFAULT_USER_INTENT["mood_style"],
        )
    if not migrated.get("shot_type"):
        old_usage = str(migrated.get("usage_type", "")).strip()
        old_lens = str(migrated.get("lens_preference", "")).strip()
        migrated["shot_type"] = legacy_shot_map.get(
            old_usage,
            legacy_shot_map.get(old_lens, DEFAULT_USER_INTENT["shot_type"]),
        )

    normalized: Dict[str, str] = {}
    for key, default_value in DEFAULT_USER_INTENT.items():
        requested = str(migrated.get(key, "")).strip()
        allowed = OPTIMIZATION_OPTION_MAP[key]
        normalized[key] = requested if requested in allowed else default_value
    return normalized


def intent_option_label(key: str, value: str) -> str:
    option = OPTIMIZATION_OPTION_MAP.get(key, {}).get(value, {})
    return str(option.get("label", value))


OUTPUT_SIZE_PRESETS: Dict[str, Dict[str, Any]] = {
    "source": {
        "dashscope_size": None,
        "doubao_size": "2K",
        "expected_ratio": None,
    },
    "screen_4_3": {
        "dashscope_size": "2048*1536",
        # Seedream 5.0 要求图片至少 3686400 像素（1920x1920），故用 2560x1920
        "doubao_size": "2560x1920",
        "expected_ratio": 4.0 / 3.0,
    },
    "portrait_3_4": {
        "dashscope_size": "1536*2048",
        "doubao_size": "1920x2560",
        "expected_ratio": 3.0 / 4.0,
    },
    "square_1_1": {
        "dashscope_size": "2048*2048",
        "doubao_size": "2048x2048",
        "expected_ratio": 1.0,
    },
    "landscape_16_9": {
        "dashscope_size": "2048*1152",
        "doubao_size": "2560x1440",
        "expected_ratio": 16.0 / 9.0,
    },
}
