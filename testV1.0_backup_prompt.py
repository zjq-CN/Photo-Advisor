import os
import re
import json
import ipaddress
import uuid
import time
import hmac
import hashlib
import base64
import subprocess
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading
import webbrowser
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
import requests
import qrcode
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

from geometry import extract_geometry  # 几何结构层：mediapipe 缺失时自动降级为 None

try:
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.hunyuan.v20230901 import hunyuan_client, models as hunyuan_models
    TENCENT_SDK_AVAILABLE = True
except ImportError:
    TENCENT_SDK_AVAILABLE = False

# =========================
# 基础配置
# =========================
# 所有本地路径都锚定主程序目录，避免从上级目录或快捷方式启动时，
# 上传图、会话和配置文件被写到意外位置。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")
LATEST_RESULT_JSON = os.path.join(STATIC_FOLDER, "latest_result.json")
SESSION_FOLDER = os.path.join(STATIC_FOLDER, "sessions")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_CONTENT_LENGTH = 15 * 1024 * 1024  # 15MB
API_KEYS_FILE = os.path.join(BASE_DIR, "_config", "api_keys.json")


def load_api_keys_file() -> Dict[str, str]:
    if not os.path.exists(API_KEYS_FILE):
        return {}
    try:
        with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            cleaned: Dict[str, str] = {}
            for k, v in data.items():
                if isinstance(v, str):
                    cleaned[str(k)] = v.strip()
            return cleaned
    except Exception:
        return {}
    return {}


def update_api_keys_file(updates: Dict[str, str]) -> None:
    current: Dict[str, Any] = {}
    if os.path.exists(API_KEYS_FILE):
        try:
            with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                current = loaded
        except Exception:
            current = {}

    for key, value in updates.items():
        current[str(key)] = str(value or "").strip()

    with open(API_KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)


def load_env(name: str, default: str = "") -> str:
    # 每次调用都重新读取文件，确保配置即时生效
    current_keys = load_api_keys_file()
    file_val = str(current_keys.get(name, "")).strip()
    if file_val:
        return file_val
    val = os.getenv(name, "").strip()
    if val:
        return val
    try:
        cmd = f"[System.Environment]::GetEnvironmentVariable('{name}','User')"
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=2.5
        )
        win_user_val = (out.stdout or "").strip()
        if win_user_val:
            return win_user_val
    except Exception:
        pass
    return default.strip()


def parse_bool(text: str, default: bool = False) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_int(text: str, default: int) -> int:
    try:
        return int(str(text or "").strip())
    except Exception:
        return default


def parse_float(text: str, default: float) -> float:
    try:
        return float(str(text or "").strip())
    except Exception:
        return default


def is_valid_ipv4(text: str) -> bool:
    try:
        ipaddress.IPv4Address(str(text or "").strip())
        return True
    except Exception:
        return False


def get_runtime_esp_config() -> Dict[str, Any]:
    enabled = parse_bool(
        load_env("ESP_SCREEN_ENABLED", "1" if ESP_SCREEN_ENABLED else "0"),
        ESP_SCREEN_ENABLED
    )
    return {
        "enabled": enabled,
        "ip": str(load_env("ESP_SCREEN_IP", ESP_SCREEN_IP)).strip(),
        "port": parse_int(load_env("ESP_SCREEN_PORT", str(ESP_SCREEN_PORT)), ESP_SCREEN_PORT),
        "endpoint": str(load_env("ESP_SCREEN_ENDPOINT", ESP_SCREEN_ENDPOINT)).strip() or "/img",
        "size": str(load_env("ESP_SCREEN_SIZE", ESP_SCREEN_SIZE)).strip() or "320x240",
        "jpeg_quality": parse_int(
            load_env("ESP_SCREEN_JPEG_QUALITY", str(ESP_SCREEN_JPEG_QUALITY)),
            ESP_SCREEN_JPEG_QUALITY
        ),
        "timeout_sec": parse_float(
            load_env("ESP_SCREEN_TIMEOUT_SEC", str(ESP_SCREEN_TIMEOUT_SEC)),
            ESP_SCREEN_TIMEOUT_SEC
        ),
    }


# =========================
# 阿里云百炼配置
# =========================
DASHSCOPE_API_KEY = load_env("DASHSCOPE_API_KEY")

# OpenAI 兼容 Chat 接口
COMPAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
COMPAT_CHAT_URL = f"{COMPAT_BASE_URL}/chat/completions"

# 百炼多模态图像生成/编辑原生接口
MULTIMODAL_GEN_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# 百炼图像模型 ID（允许通过 api_keys.json /环境变量覆盖）
WAN_IMAGE_MODEL = load_env("WAN_IMAGE_MODEL", "wan2.7-image")
QWEN_IMAGE_MODEL = load_env("QWEN_IMAGE_MODEL", "qwen-image-2.0")
QWEN_IMAGE_EDIT_MAX_MODEL = load_env("QWEN_IMAGE_EDIT_MAX_MODEL", "qwen-image-edit-max")

# 智谱图像配置
ZHIPU_API_KEY = load_env("ZHIPU_API_KEY")
ZHIPU_IMAGE_URL = load_env("ZHIPU_IMAGE_URL", "https://open.bigmodel.cn/api/paas/v4/images/generations")
ZHIPU_IMAGE_MODEL = load_env("ZHIPU_IMAGE_MODEL", "glm-image")
ZHIPU_CHAT_URL = load_env("ZHIPU_CHAT_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
ZHIPU_VISION_MODEL = load_env("ZHIPU_VISION_MODEL", "glm-4.6v-flash")

# 豆包图像配置
DOUBAO_API_KEY = load_env("DOUBAO_API_KEY")
DOUBAO_IMAGE_URL = load_env("DOUBAO_IMAGE_URL", "https://ark.cn-beijing.volces.com/api/v3/images/generations")
DOUBAO_IMAGE_MODEL = load_env("DOUBAO_IMAGE_MODEL", "doubao-seedream-5-0-260128")
DOUBAO_IMAGE_FALLBACK_MODEL = load_env(
    "DOUBAO_IMAGE_FALLBACK_MODEL",
    "doubao-seedream-5-0-lite-260128",
)
DOUBAO_MODEL_LIST_URL = load_env("DOUBAO_MODEL_LIST_URL", "https://ark.cn-beijing.volces.com/api/v3/models")
DOUBAO_ENDPOINT_ID = load_env("DOUBAO_ENDPOINT_ID")

# 即梦图像配置
JIMENG_ACCESS_KEY_ID = load_env("JIMENG_ACCESS_KEY_ID")
JIMENG_SECRET_ACCESS_KEY = load_env("JIMENG_SECRET_ACCESS_KEY")
JIMENG_IMAGE_URL = load_env("JIMENG_IMAGE_URL")
JIMENG_IMAGE_MODEL = load_env("JIMENG_IMAGE_MODEL", "jimeng-ai-image-4.0")
JIMENG_REQ_KEY = load_env("JIMENG_REQ_KEY", "jimeng_t2i_v40")

HUNYUAN_API_KEY = load_env("HUNYUAN_API_KEY")
HUNYUAN_SECRET_ID = load_env("HUNYUAN_SECRET_ID") # For SDK
HUNYUAN_SECRET_KEY = load_env("HUNYUAN_SECRET_KEY") # For SDK
HUNYUAN_CHAT_URL = load_env("HUNYUAN_CHAT_URL", "https://tokenhub.tencentmaas.com/v1/chat/completions")
HUNYUAN_VISION_MODEL = load_env("HUNYUAN_VISION_MODEL", "hy-vision-2.0-instruct")

# DeepSeek 视觉理解配置
DEEPSEEK_API_KEY = load_env("DEEPSEEK_API_KEY")
DEEPSEEK_CHAT_URL = load_env("DEEPSEEK_CHAT_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_VISION_MODEL = load_env("DEEPSEEK_VISION_MODEL", "deepseek-v4-flash-vision-exp")

# ESP32 屏幕推送配置
ESP_SCREEN_ENABLED = parse_bool(load_env("ESP_SCREEN_ENABLED", "1"), True)
ESP_SCREEN_IP = load_env("ESP_SCREEN_IP")
ESP_SCREEN_PORT = parse_int(load_env("ESP_SCREEN_PORT", "80"), 80)
ESP_SCREEN_ENDPOINT = load_env("ESP_SCREEN_ENDPOINT", "/img")
ESP_SCREEN_SIZE = load_env("ESP_SCREEN_SIZE", "320x240")
ESP_SCREEN_JPEG_QUALITY = parse_int(load_env("ESP_SCREEN_JPEG_QUALITY", "85"), 85)
ESP_SCREEN_TIMEOUT_SEC = parse_float(load_env("ESP_SCREEN_TIMEOUT_SEC", "6"), 6.0)

# 视觉理解模型：主 -> 备（按 2026-08 模型考试结果排序）
# qwen3.5-flash 经测试选为唯一主模型：6/6 成功、平均 12.7s、分析质量稳定；
# 其余按稳定性/速度作 fallback，同一供应商（阿里百炼）的 qwen 模型错开，
# 避免百炼整体限流时连续失败。API Key 配置保留在 _config/api_keys.json，
# 换模型只需调整本列表，无需改 Key 文件。
VISION_MODELS = [
    "qwen3.5-flash",
    "deepseek-v4-flash-vision-exp",
    "qwen3-vl-plus",
    "hy-vision-2.0-instruct",
    "glm-4.6v-flash",
]

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

app = Flask(
    __name__,
    template_folder=TEMPLATE_FOLDER,
    static_folder=STATIC_FOLDER,
    static_url_path="/static",
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)


# =========================
# 通用工具
# =========================
def save_latest_result(payload: dict) -> None:
    with open(LATEST_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_session_id(session_id: str) -> str:
    clean = str(session_id or "").strip()
    if not SESSION_ID_PATTERN.fullmatch(clean):
        raise ValueError("session_id 格式不正确")
    return clean


def build_upload_filename(raw_filename: str) -> str:
    if "." in raw_filename:
        ext = raw_filename.rsplit(".", 1)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError("不支持的图片格式，仅支持 jpg/jpeg/png/webp")
    else:
        ext = "jpg"

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"photo_{now_str}_{unique_id}.{ext}"


def save_session_data(session_id: str, payload: Dict[str, Any]) -> None:
    safe_session_id = validate_session_id(session_id)
    session_path = os.path.join(SESSION_FOLDER, f"{safe_session_id}.json")
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_session_data(session_id: str) -> Dict[str, Any]:
    safe_session_id = validate_session_id(session_id)
    session_path = os.path.join(SESSION_FOLDER, f"{safe_session_id}.json")
    if not os.path.exists(session_path):
        raise FileNotFoundError("未找到对应会话，请先拍照诊断。")
    with open(session_path, "r", encoding="utf-8") as f:
        return json.load(f)


def open_browser_later(url: str, delay: float = 1.2) -> None:
    """
    Flask 启动后稍等片刻，再自动打开默认浏览器
    """
    def _open():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            print(f"[自动打开] {url}")
        except Exception as e:
            print(f"[自动打开失败] {e}")

    threading.Thread(target=_open, daemon=True).start()

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_file_to_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime = mime_map.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def image_file_to_jimeng_base64(image_path: str, max_edge: int = 1024, jpeg_quality: int = 88) -> str:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise RuntimeError("即梦输入图像读取失败")
    h, w = img.shape[:2]
    scale = min(1.0, float(max_edge) / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("即梦输入图像编码失败")
    return base64.b64encode(enc.tobytes()).decode("utf-8")


def extract_json_from_text(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    brace = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(1))

    raise ValueError("未能从模型返回中解析出 JSON")


def safe_resp_text(resp: requests.Response) -> str:
    try:
        return resp.text
    except Exception:
        return ""


def is_busy_or_rate_limited(resp: requests.Response) -> bool:
    text = safe_resp_text(resp).lower()
    if resp.status_code == 429:
        return True
    keywords = [
        "rate limit",
        "too many requests",
        "当前访问量过大",
        "稍后再试",
        "busy",
        "quota",
        "throttle",
    ]
    return any(k.lower() in text for k in keywords)


def parse_qwen_image_edit_url(data: Dict[str, Any]) -> str:
    content = (
        data.get("output", {})
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", [])
    )

    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("image"):
                return item["image"]

    raise RuntimeError("Qwen-Image-Edit 未返回 image URL。")


def parse_image_url_any(data: Any) -> Optional[str]:
    if isinstance(data, dict):
        for key in ("url", "image", "image_url"):
            val = data.get(key)
            if isinstance(val, str) and (
                val.startswith("http://")
                or val.startswith("https://")
                or val.startswith("/uploads/")
            ):
                return val
        for v in data.values():
            hit = parse_image_url_any(v)
            if hit:
                return hit
    elif isinstance(data, list):
        for item in data:
            hit = parse_image_url_any(item)
            if hit:
                return hit
    return None


def parse_image_base64_any(data: Any) -> Optional[str]:
    if isinstance(data, dict):
        # 增加 binary_data_base64 (Jimeng)
        for key in ("b64_json", "image_base64", "base64", "b64", "image_data", "binary_data_base64"):
            val = data.get(key)
            if isinstance(val, str) and len(val.strip()) > 80:
                return val.strip()
            # Handle list of base64 strings (Jimeng returns binary_data_base64 as list)
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], str) and len(val[0]) > 80:
                return val[0].strip()
        
        for v in data.values():
            hit = parse_image_base64_any(v)
            if hit:
                return hit
    elif isinstance(data, list):
        for item in data:
            hit = parse_image_base64_any(item)
            if hit:
                return hit
    return None


def parse_task_id_any(data: Any) -> Optional[str]:
    if isinstance(data, dict):
        for key in ("task_id", "taskId", "id"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        for v in data.values():
            hit = parse_task_id_any(v)
            if hit:
                return hit
    elif isinstance(data, list):
        for item in data:
            hit = parse_task_id_any(item)
            if hit:
                return hit
    return None


def normalize_doubao_url(url: str) -> str:
    clean = str(url or "").strip()
    if not clean:
        clean = "https://ark.cn-beijing.volces.com/api/v3/images/edits"
    if not clean.startswith("http://") and not clean.startswith("https://"):
        clean = f"https://{clean.lstrip('/')}"
    clean = clean.replace("ark-cn-beijing.volces.com", "ark.cn-beijing.volces.com")
    parsed = urlparse(clean)
    path = parsed.path or "/"
    if path in ("", "/"):
        path = "/api/v3/images/edits"
    if path.endswith("/images"):
        path = f"{path}/edits"
    if "/api/v3/images/" not in path and not path.endswith("/models"):
        if path.endswith("/edits"):
            path = "/api/v3/images/edits"
        elif path.endswith("/generations"):
            path = "/api/v3/images/generations"
        else:
            path = "/api/v3/images/edits"
    return urlunparse((parsed.scheme or "https", parsed.netloc, path, "", parsed.query, ""))


def build_jimeng_req_key_candidates(req_key: str) -> List[str]:
    raw_candidates = [
        str(req_key or "").strip(),
        "jimeng_t2i_v40",
        "t2i_v40_jimeng",
    ]
    seen: set = set()
    candidates: List[str] = []
    for item in raw_candidates:
        if not item or item in seen:
            continue
        candidates.append(item)
        seen.add(item)
    return candidates


def list_doubao_model_ids(timeout: int = 20) -> List[str]:
    if not DOUBAO_API_KEY:
        return []
    url = str(DOUBAO_MODEL_LIST_URL or "").strip()
    if not url:
        return []
    headers = {
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except Exception:
        return []
    if resp.status_code != 200:
        return []
    try:
        data = resp.json()
    except Exception:
        return []
    items = data.get("data", [])
    if not isinstance(items, list):
        return []
    model_ids: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if isinstance(mid, str) and mid.strip():
            model_ids.append(mid.strip())
    return model_ids


def build_doubao_model_candidates() -> List[str]:
    # 先试 Seedream 5.0 完整版，仅在账号未开通/服务不可用时回退 Lite。
    # 不再将账号下其他图像模型混入候选，避免界面显示一个模型却实际调用另一个。
    configured_model = str(DOUBAO_IMAGE_MODEL or "").strip()
    fallback_model = str(DOUBAO_IMAGE_FALLBACK_MODEL or "").strip()
    configured_endpoint = str(DOUBAO_ENDPOINT_ID or "").strip()
    preferred_candidates = [
        "doubao-seedream-5-0-260128",
        configured_model,
        configured_model.lower() if configured_model else "",
        configured_endpoint,
        fallback_model,
        fallback_model.lower() if fallback_model else "",
        "doubao-seedream-5-0-lite-260128",
    ]

    fetched_ids = list_doubao_model_ids()
    seedream5_ids = [mid for mid in fetched_ids if "seedream-5-0" in mid.lower()]

    seen: set = set()
    ordered: List[str] = []
    for item in preferred_candidates + seedream5_ids:
        normalized = str(item or "").strip()
        dedupe_key = normalized if normalized.lower().startswith("ep-") else normalized.lower()
        if not normalized or dedupe_key in seen:
            continue
        ordered.append(normalized)
        seen.add(dedupe_key)
    return ordered


def sha256_hex(content: bytes) -> str:
    h = hashlib.sha256()
    h.update(content)
    return h.hexdigest()


def hmac_sha256(key: bytes, content: str) -> bytes:
    return hmac.new(key, content.encode("utf-8"), hashlib.sha256).digest()


def build_volc_signed_headers(
    method: str,
    endpoint: str,
    query: Dict[str, str],
    body_bytes: bytes,
    access_key_id: str,
    secret_access_key: str,
    region: str = "cn-north-1",
    service: str = "cv",
) -> Dict[str, str]:
    parsed = urlparse(endpoint)
    host = parsed.netloc
    path = parsed.path or "/"
    x_date = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    short_date = x_date[:8]
    canonical_query = urlencode(sorted([(k, v) for k, v in query.items()]), doseq=True)
    x_content_sha256 = sha256_hex(body_bytes)
    signed_headers = "content-type;host;region;service;x-content-sha256;x-date"
    canonical_headers = (
        f"content-type:application/json\n"
        f"host:{host}\n"
        f"region:{region}\n"
        f"service:{service}\n"
        f"x-content-sha256:{x_content_sha256}\n"
        f"x-date:{x_date}\n"
    )
    canonical_request = "\n".join([
        method.upper(),
        path,
        canonical_query,
        canonical_headers,
        signed_headers,
        x_content_sha256,
    ])
    credential_scope = f"{short_date}/{region}/{service}/request"
    string_to_sign = "\n".join([
        "HMAC-SHA256",
        x_date,
        credential_scope,
        sha256_hex(canonical_request.encode("utf-8")),
    ])
    k_date = hmac_sha256(secret_access_key.encode("utf-8"), short_date)
    k_region = hmac_sha256(k_date, region)
    k_service = hmac_sha256(k_region, service)
    k_signing = hmac_sha256(k_service, "request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = (
        f"HMAC-SHA256 Credential={access_key_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return {
        "Content-Type": "application/json",
        "Region": region,
        "Service": service,
        "X-Date": x_date,
        "X-Content-Sha256": x_content_sha256,
        "Authorization": authorization,
    }


def post_jimeng_signed(url: str, body: Dict[str, Any], action: str, version: str = "2022-08-31", timeout: int = 180) -> requests.Response:
    parsed = urlparse(url)
    endpoint = urlunparse((parsed.scheme or "https", parsed.netloc, parsed.path or "/", "", "", ""))
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    q["Action"] = action
    q["Version"] = version
    body_text = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    body_bytes = body_text.encode("utf-8")
    headers = build_volc_signed_headers(
        method="POST",
        endpoint=endpoint,
        query=q,
        body_bytes=body_bytes,
        access_key_id=JIMENG_ACCESS_KEY_ID,
        secret_access_key=JIMENG_SECRET_ACCESS_KEY,
        region="cn-north-1",
        service="cv",
    )
    return requests.post(endpoint, params=q, headers=headers, data=body_bytes, timeout=timeout)


def save_base64_image_to_uploads(raw_b64: str, prefix: str = "generated_raw") -> str:
    b64_text = str(raw_b64 or "").strip()
    if not b64_text:
        raise RuntimeError("base64 图片内容为空")
    ext = "png"
    if b64_text.startswith("data:image/"):
        head, _, body = b64_text.partition(",")
        mime_part = head.split(";")[0]
        ext_part = mime_part.split("/")[-1].lower()
        if ext_part in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg" if ext_part == "jpeg" else ext_part
        b64_text = body.strip()
    image_bytes = base64.b64decode(b64_text, validate=False)
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}"
    local_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(local_path, "wb") as f:
        f.write(image_bytes)
    return f"/uploads/{filename}"


def load_image_from_url_for_quality(url: str) -> Optional[np.ndarray]:
    try:
        if not isinstance(url, str) or not url.strip():
            return None
        url = url.strip()
        if url.startswith("/uploads/"):
            local_name = url.split("/uploads/", 1)[1]
            local_path = os.path.join(UPLOAD_FOLDER, local_name)
            if not os.path.exists(local_path):
                return None
            return cv2.imread(local_path)
        if not (url.startswith("http://") or url.startswith("https://")):
            return None
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return None
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def parse_screen_size(text: str, default_w: int = 320, default_h: int = 240) -> Tuple[int, int]:
    raw = str(text or "").strip().lower()
    if "x" not in raw:
        return default_w, default_h
    left, right = raw.split("x", 1)
    width = parse_int(left, default_w)
    height = parse_int(right, default_h)
    if width <= 0 or height <= 0:
        return default_w, default_h
    return width, height


def resolve_image_ref_to_bytes(image_ref: str) -> bytes:
    ref = str(image_ref or "").strip()
    if not ref:
        raise RuntimeError("图片地址为空，无法推送到屏幕")

    if ref.startswith("/uploads/"):
        local_name = ref.split("/uploads/", 1)[1]
        local_path = os.path.join(UPLOAD_FOLDER, local_name)
        if not os.path.exists(local_path):
            raise RuntimeError(f"本地图片不存在：{local_path}")
        with open(local_path, "rb") as f:
            return f.read()

    if ref.startswith("http://") or ref.startswith("https://"):
        resp = requests.get(ref, timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"下载图片失败，HTTP {resp.status_code}")
        return resp.content

    if os.path.exists(ref):
        with open(ref, "rb") as f:
            return f.read()

    raise RuntimeError(f"不支持的图片地址：{ref}")


def build_screen_jpeg(image_bytes: bytes, target_w: int, target_h: int, quality: int) -> Tuple[bytes, bool]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise RuntimeError("图片解码失败，无法发送到屏幕")

    src_h, src_w = img.shape[:2]
    # 不再为适配横屏而旋转竖图，否则人物会横倒。
    # 使用 contain + 居中留黑边，完整保留人物和构图；4:3 输出则会恰好铺满屏幕。
    scale = min(float(target_w) / float(src_w), float(target_h) / float(src_h))
    resized_w = max(1, min(target_w, int(round(src_w * scale))))
    resized_h = max(1, min(target_h, int(round(src_h * scale))))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (resized_w, resized_h), interpolation=interp)
    frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - resized_w) // 2
    y0 = (target_h - resized_h) // 2
    frame[y0:y0 + resized_h, x0:x0 + resized_w] = resized

    jpeg_quality = min(95, max(50, int(quality)))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败，无法发送到屏幕")
    return buf.tobytes(), False


def push_image_to_esp(image_ref: str, append: bool = False) -> Tuple[bool, str]:
    cfg = get_runtime_esp_config()
    if not cfg["enabled"]:
        return False, "ESP 推送已关闭（ESP_SCREEN_ENABLED=0）"
    if not cfg["ip"]:
        return False, "未配置 ESP_SCREEN_IP"

    width, height = parse_screen_size(str(cfg["size"]), 320, 240)
    endpoint = str(cfg["endpoint"])
    endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    url = f"http://{cfg['ip']}:{cfg['port']}{endpoint}"
    if append:
        url += "?append=1"

    try:
        raw_bytes = resolve_image_ref_to_bytes(image_ref)
        q = int(cfg["jpeg_quality"])
        jpg_bytes, rotated = build_screen_jpeg(raw_bytes, width, height, q)
        resp = requests.post(
            url,
            files={"file": ("frame.jpg", jpg_bytes, "image/jpeg")},
            timeout=max(1.0, float(cfg["timeout_sec"])),
        )
        rotate_note = "rotated90=1" if rotated else "rotated90=0"
        if resp.status_code != 200:
            body = (resp.text or "").strip()
            if resp.status_code == 400 and "STORE_FAILED" in body.upper():
                return False, (
                    f"ESP 存储失败（HTTP 400: {body[:120]}）。"
                    "请确认固件按 ESP32-S3 OPI PSRAM 模式编译（PSRAM=opi），当前图片 "
                    f"{len(jpg_bytes)} bytes, q={q}, {rotate_note}"
                )
            return False, f"ESP 推送失败 HTTP {resp.status_code}: {body[:120]} (q={q}, {len(jpg_bytes)} bytes, {rotate_note})"
        return True, f"ESP 推送成功：{url} ({len(jpg_bytes)} bytes, q={q}, {rotate_note})"
    except Exception as e:
        return False, f"ESP 推送异常：{e}"


def build_esp_push_queue(
    primary_image: str,
    model_candidates: List[Dict[str, Any]],
    include_all: bool = False,
) -> List[str]:
    queue: List[str] = []

    def _add(url_text: Any) -> None:
        url = str(url_text or "").strip()
        if not url:
            return
        if url in queue:
            return
        queue.append(url)

    _add(primary_image)
    if not include_all:
        return queue

    for item in model_candidates:
        if not isinstance(item, dict):
            continue
        if item.get("error"):
            continue
        _add(item.get("diagram_url"))
    return queue


def evaluate_image_quality(
    model_name: str,
    image_url: str,
    generation_settings: Optional[Dict[str, Any]] = None,
    user_intent: Optional[Dict[str, str]] = None,
) -> Tuple[float, str]:
    _ = model_name
    img = load_image_from_url_for_quality(image_url)
    if img is None or img.size == 0:
        return -999.0, "图片不可读取"

    height, width = img.shape[:2]
    actual_aspect = float(width) / float(max(1, height))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    col_signal = gray.mean(axis=0)
    row_signal = gray.mean(axis=1)
    col_std = float(np.std(col_signal))
    row_std = float(np.std(row_signal))
    stripe_ratio = col_std / (row_std + 1e-6)
    edges = cv2.Canny(gray, 70, 140)
    edge_density = float(np.mean(edges > 0))
    mean_luma = float(np.mean(gray))
    clipped_dark = float(np.mean(gray <= 5))
    clipped_light = float(np.mean(gray >= 250))

    # 只根据实际输出图和用户选择评分，不给任何模型预设加分。
    score = 0.0
    goal = (user_intent or {}).get("optimization_goal", "auto")

    if lap_var < 12:
        score -= 1.2
    elif lap_var > 950:
        score -= 0.8
    else:
        score += 0.4
    if goal == "clarity" and 35 <= lap_var <= 800:
        score += 0.6

    if stripe_ratio > 2.2 and edge_density > 0.07:
        score -= 4.0
        return score, f"疑似条纹伪影 ratio={stripe_ratio:.2f}, edge={edge_density:.3f}"

    if edge_density > 0.24:
        score -= 1.0
    elif edge_density < 0.02:
        score -= 0.6
    else:
        score += 0.3

    if mean_luma < 35 or mean_luma > 225:
        score -= 1.0
    elif 60 <= mean_luma <= 200:
        score += 0.25
    if clipped_dark + clipped_light > 0.20:
        score -= 0.8
    elif goal == "lighting" and clipped_dark + clipped_light < 0.08:
        score += 0.7

    expected_aspect = (generation_settings or {}).get("expected_ratio")
    aspect_error = 0.0
    if isinstance(expected_aspect, (int, float)) and expected_aspect > 0:
        aspect_error = abs(actual_aspect - float(expected_aspect)) / float(expected_aspect)
        if aspect_error <= 0.03:
            score += 1.0
        elif aspect_error <= 0.08:
            score += 0.25
        else:
            score -= min(2.0, aspect_error * 4.0)

    return score, (
        f"ok artifact={stripe_ratio:.2f}, edge={edge_density:.3f}, lap={lap_var:.1f}, "
        f"luma={mean_luma:.1f}, aspect={actual_aspect:.3f}, aspect_err={aspect_error:.3f}"
    )


# =========================
# Windows ipconfig 提取局域网IP
# =========================
def get_local_ip() -> str:
    """
    从中英文 Windows ipconfig 输出中提取局域网 IPv4。
    优先无线网卡，其次物理以太网，最后才使用其他非虚拟适配器。
    """
    try:
        result = subprocess.run(
            ["ipconfig"],
            capture_output=True,
            text=True,
            encoding="gbk",
            errors="ignore"
        )
        text = result.stdout.replace("\r\n", "\n")
        # 直接在中英文 adapter 标题前切分；不能按空行切，因为部分 Windows
        # 版本会在网卡标题和字段之间也插入空行。
        blocks = re.split(
            r"(?im)(?=^[^\n]*(?:adapter|适配器)[^:\n]*:\s*$)",
            text,
        )

        ethernet_ip: Optional[str] = None
        fallback_ip: Optional[str] = None
        virtual_markers = (
            "virtual", "vethernet", "vmware", "hyper-v", "bluetooth",
            "loopback", "tunnel", "虚拟", "隧道", "蓝牙",
        )

        for block in blocks:
            block_strip = block.strip()
            if not block_strip:
                continue

            title = block_strip.split("\n")[0].lower()

            m = re.search(r"IPv4[^:\n]*:\s*([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)", block, re.IGNORECASE)
            if not m:
                continue

            ip = m.group(1)
            if ip.startswith("169.254.") or ip == "127.0.0.1":
                continue

            if (
                "无线局域网适配器" in title
                or "wireless lan" in title
                or "wlan" in title
                or "wi-fi" in title
                or "wifi" in title
            ):
                print(f"[IP识别] 选择无线网卡: {title}")
                print(f"[IP识别] IPv4: {ip}")
                return ip

            is_virtual = any(marker in title for marker in virtual_markers)
            if not is_virtual and ("以太网适配器" in title or "ethernet adapter" in title):
                ethernet_ip = ethernet_ip or ip
            elif not is_virtual:
                fallback_ip = fallback_ip or ip

        if ethernet_ip:
            print("[IP识别] 未找到无线网卡，改用以太网 IPv4")
            print(f"[IP识别] IPv4: {ethernet_ip}")
            return ethernet_ip

        if fallback_ip:
            print("[IP识别] 使用其他活动网卡 IPv4")
            print(f"[IP识别] IPv4: {fallback_ip}")
            return fallback_ip

    except Exception as e:
        print(f"[IP识别] 解析 ipconfig 失败: {e}")

    return "127.0.0.1"


def generate_qr_image(text: str, save_path: str) -> None:
    qr = qrcode.QRCode(
        version=1,
        box_size=10,
        border=2,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(save_path)


def generate_access_qrcodes(host_ip: str, port: int = 5000) -> Dict[str, str]:
    mobile_url = f"http://{host_ip}:{port}/mobile"

    mobile_qr_path = os.path.join(STATIC_FOLDER, "qr_mobile.png")
    generate_qr_image(mobile_url, mobile_qr_path)

    return {
        "mobile_url": mobile_url,
        "mobile_qr": "/static/qr_mobile.png",
    }


# =========================
# 第一次分析提示词
# =========================
def build_analysis_prompt(geometry_block: str = "") -> str:
    prompt = """
你是专业摄影指导。分析照片后，只输出一个 JSON 对象（不要 markdown 代码块，不要任何解释文字），key 必须如下：

{
  "scene_summary": "画面简短概括，1-2句",
  "subject_gender": "仅填\"男性\"或\"女性\"，按照片真实外观",
  "subject_position_analysis": "人物位置（偏左/右/居中、高低、留白是否合理）",
  "camera_angle_analysis": "拍摄角度（平拍/俯拍/仰拍、正面/侧面）及是否合适",
  "shot_size_analysis": "景别（头像/半身/全身/环境人像）、人物占比、裁切是否合理",
  "composition_analysis": "构图诊断：头顶留白、人物偏移、前景遮挡、水平垂直线、背景干扰",
  "light_source_inference": "主光源方向及依据；若明暗关系不明显、无法可靠判断，写\"光源方向不明显\"，不要强行推测",
  "recommended_shooting_position": "摄影师下一步机位与站位建议",
  "suggested_adjustment": "人物姿态与相机分别如何调整",
  "ideal_image_prompt": "按建议调整完成后的理想画面描述（见下方要求）"
}

ideal_image_prompt 要求：
1. 描述\"调整后\"的理想画面，不是当前照片，不是建议或教学说明；
2. 重点描述人物理想位置、理想角度、整体观感，不详细描述环境（编辑模型会参考原图）；
3. 仅在能明确判断光源方向时描述光线，否则不强调具体光源位置；
4. 真实照片效果，非卡通/插画，无箭头图标标注文字，人物性别与 subject_gender 一致。

整体：输出中文；不虚构原图中没有的光源；分析建议要可执行。
""".strip()
    if geometry_block:
        return prompt + "\n\n" + geometry_block
    return prompt

def normalize_ai_result(result: Dict[str, Any]) -> Dict[str, str]:
    keys = [
        "scene_summary",
        "subject_gender",
        "subject_position_analysis",
        "camera_angle_analysis",
        "shot_size_analysis",
        "composition_analysis",
        "light_source_inference",
        "recommended_shooting_position",
        "suggested_adjustment",
        "ideal_image_prompt",
    ]
    normalized: Dict[str, str] = {}
    for k in keys:
        v = result.get(k, "")
        normalized[k] = v if isinstance(v, str) and v.strip() else "未返回该项结果。"
    return normalized


# =========================
# 视觉理解
# =========================
def call_vision_once(model_name: str, image_path: str, geometry_block: str = "") -> Dict[str, str]:
    if model_name.startswith("hunyuan") or model_name.startswith("hy-"):
        return call_vision_hunyuan_once(model_name, image_path, geometry_block)
    if model_name.startswith("deepseek"):
        return call_vision_deepseek_once(model_name, image_path, geometry_block)
    if model_name.startswith("glm-"):
        return call_vision_zhipu_once(model_name, image_path, geometry_block)
    image_data_url = image_file_to_data_url(image_path)
    prompt = build_analysis_prompt(geometry_block)

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "max_tokens": 2048,
        "extra_body": {
            "enable_thinking": False
        }
    }

    resp = requests.post(COMPAT_CHAT_URL, headers=headers, json=payload, timeout=120)

    print(f"=== 视觉模型 {model_name} 状态码 ===")
    print(resp.status_code)
    print(resp.text[:3000])

    if resp.status_code != 200:
        raise requests.HTTPError(response=resp)

    data = resp.json()
    message = data["choices"][0]["message"]
    content = message.get("content", "")

    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        content_text = "\n".join(text_parts).strip()
    else:
        content_text = str(content).strip()

    result = extract_json_from_text(content_text)
    return normalize_ai_result(result)


def call_vision_hunyuan_once(model_name: str, image_path: str, geometry_block: str = "") -> Dict[str, str]:
    # 检查 Key 格式
    if HUNYUAN_API_KEY and HUNYUAN_API_KEY.startswith("AKID"):
        print("!!! 警告: HUNYUAN_API_KEY 看起来像是腾讯云 SecretId (以 AKID 开头)。")
        print("!!! 混元 OpenAI 兼容接口需要单独的 API Key，而非 SecretId/SecretKey。")
        print("!!! 请前往 https://console.cloud.tencent.com/hunyuan/start 获取 API Key。")

    image_data_url = image_file_to_data_url(image_path)
    prompt = build_analysis_prompt(geometry_block)
    headers = {
        "Authorization": f"Bearer {HUNYUAN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name if model_name.strip() else HUNYUAN_VISION_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    }
    resp = requests.post(HUNYUAN_CHAT_URL, headers=headers, json=payload, timeout=120)
    print(f"=== 视觉模型 {model_name} 状态码 ===")
    print(resp.status_code)
    print(resp.text[:3000])
    if resp.status_code != 200:
        raise requests.HTTPError(response=resp)
    data = resp.json()
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
        content_text = "\n".join(text_parts).strip()
    else:
        content_text = str(content).strip()
    result = extract_json_from_text(content_text)
    return normalize_ai_result(result)


def call_vision_deepseek_once(model_name: str, image_path: str, geometry_block: str = "") -> Dict[str, str]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")

    image_data_url = image_file_to_data_url(image_path)
    prompt = build_analysis_prompt(geometry_block)
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name if model_name.strip() else DEEPSEEK_VISION_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "max_tokens": 2048,
    }
    resp = requests.post(DEEPSEEK_CHAT_URL, headers=headers, json=payload, timeout=120)
    print(f"=== 视觉模型 {model_name} 状态码 ===")
    print(resp.status_code)
    print(resp.text[:3000])
    if resp.status_code != 200:
        raise requests.HTTPError(response=resp)
    data = resp.json()
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
        content_text = "\n".join(text_parts).strip()
    else:
        content_text = str(content).strip()
    result = extract_json_from_text(content_text)
    return normalize_ai_result(result)


def call_vision_zhipu_once(model_name: str, image_path: str, geometry_block: str = "") -> Dict[str, str]:
    if not ZHIPU_API_KEY:
        raise RuntimeError("未配置 ZHIPU_API_KEY")

    image_data_url = image_file_to_data_url(image_path)
    prompt = build_analysis_prompt(geometry_block)
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name if model_name.strip() else ZHIPU_VISION_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "max_tokens": 2048,
    }
    resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=120)
    print(f"=== 视觉模型 {model_name} 状态码 ===")
    print(resp.status_code)
    print(resp.text[:3000])
    if resp.status_code != 200:
        raise requests.HTTPError(response=resp)
    data = resp.json()
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
        content_text = "\n".join(text_parts).strip()
    else:
        content_text = str(content).strip()
    result = extract_json_from_text(content_text)
    return normalize_ai_result(result)


def check_vision_model_ready(model_name: str) -> Tuple[bool, str]:
    if model_name.startswith("hunyuan") or model_name.startswith("hy-"):
        return bool(HUNYUAN_API_KEY), "未配置 HUNYUAN_API_KEY"
    if model_name.startswith("deepseek"):
        return bool(DEEPSEEK_API_KEY), "未配置 DEEPSEEK_API_KEY"
    if model_name.startswith("glm-"):
        return bool(ZHIPU_API_KEY), "未配置 ZHIPU_API_KEY"
    return bool(DASHSCOPE_API_KEY), "未配置 DASHSCOPE_API_KEY"


def call_vision_auto(image_path: str, geometry_block: str = "") -> Tuple[Dict[str, str], str]:
    last_error: Optional[Exception] = None
    ready_models: List[str] = []
    for model_name in VISION_MODELS:
        ready, _ = check_vision_model_ready(model_name)
        if not ready:
            continue
        ready_models.append(model_name)
        try:
            result = call_vision_once(model_name, image_path, geometry_block)
            return result, model_name
        except requests.HTTPError as e:
            last_error = e
            resp = e.response
            if resp is not None and is_busy_or_rate_limited(resp):
                print(f"[视觉模型繁忙/限流] {model_name} 失败，切换下一个模型")
                continue
            print(f"[视觉模型失败] {model_name} 返回非 200，切换下一个模型")
            continue
        except Exception as e:
            last_error = e
            print(f"[视觉模型异常] {model_name}: {e}")
            continue

    if not ready_models:
        raise RuntimeError("未配置可用视觉模型密钥。")
    raise last_error if last_error else RuntimeError("所有视觉理解模型都调用失败。")


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


def build_strategy_plan(ai_result: Dict[str, str], user_intent: Dict[str, str]) -> Dict[str, str]:
    requested_goal = user_intent["optimization_goal"]
    effective_goal = infer_automatic_goal(ai_result) if requested_goal == "auto" else requested_goal

    strategy: Dict[str, str] = {
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

    goal_updates: Dict[str, Dict[str, str]] = {
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
    strategy.update(goal_updates[effective_goal])

    shot_updates: Dict[str, Dict[str, str]] = {
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
    strategy.update(shot_updates[user_intent["shot_type"]])

    mood_updates = {
        "keep": "除完成主目标所需的轻微纠偏外，保持原图色温、饱和度和对比关系",
        "natural": "自然中性、通透但不过曝，肤色真实",
        "warm": "在原图基础上轻微增暖，肤色不偏黄不偏红",
        "fresh": "明度轻盈、色彩干净，保留高光细节",
        "cool": "轻微冷调与克制饱和度，人物肤色仍保持真实",
        "cinematic": "使用原图已有明暗形成克制的电影对比，不新增光源或色彩特效",
    }
    mood_strategy = mood_updates[user_intent["mood_style"]]
    if effective_goal == "lighting":
        # 光线主目标需要保留白平衡/肤色修复要求；氛围只决定调色方向，
        # 不能像旧版那样把光线策略整体覆盖掉。
        strategy["color_strategy"] = f"{strategy['color_strategy']}；氛围控制：{mood_strategy}"
    else:
        strategy["color_strategy"] = mood_strategy

    strength_updates = {
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
    strategy["strength_rule"] = strength_updates[user_intent["edit_strength"]]

    light_text = ai_result.get("light_source_inference", "")
    if "不明显" in light_text or "无法" in light_text:
        strategy["light_strategy"] += " 原图光源方向不可靠，不制造新的窗光、轮廓光或逆光。"

    return strategy


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


def edit_with_dashscope_image_model(
    image_path: str,
    prompt_text: str,
    model_id: str,
    use_negative_prompt: bool,
    generation_settings: Optional[Dict[str, Any]] = None,
) -> str:
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY")
    image_data_url = image_file_to_data_url(image_path)

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    parameters: Dict[str, Any] = {
        "n": 1,
        "watermark": False,
    }
    requested_size = (generation_settings or {}).get("dashscope_size")
    if requested_size:
        parameters["size"] = requested_size
    if use_negative_prompt:
        parameters["negative_prompt"] = QWEN_IMAGE_NEGATIVE_PROMPT
        # 本项目的提示词已由诊断结果精确组装，不再让模型二次改写。
        parameters["prompt_extend"] = False

    payload = {
        "model": model_id,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": image_data_url},
                        {"text": prompt_text},
                    ],
                }
            ]
        },
        "parameters": parameters,
    }

    resp = requests.post(MULTIMODAL_GEN_URL, headers=headers, json=payload, timeout=180)

    print(f"=== {model_id} 状态码 ===")
    print(resp.status_code)
    print(resp.text[:3000])

    if resp.status_code != 200:
        raise requests.HTTPError(response=resp)

    data = resp.json()
    image_url = parse_image_url_any(data)
    if not image_url:
        raise RuntimeError(f"{model_id} 未返回图片 URL。")
    return image_url


def edit_with_wan27_image(
    image_path: str,
    prompt_text: str,
    generation_settings: Optional[Dict[str, Any]] = None,
) -> str:
    return edit_with_dashscope_image_model(
        image_path,
        prompt_text,
        WAN_IMAGE_MODEL,
        use_negative_prompt=False,
        generation_settings=generation_settings,
    )


def edit_with_qwen_image_20(
    image_path: str,
    prompt_text: str,
    generation_settings: Optional[Dict[str, Any]] = None,
) -> str:
    return edit_with_dashscope_image_model(
        image_path,
        prompt_text,
        QWEN_IMAGE_MODEL,
        use_negative_prompt=True,
        generation_settings=generation_settings,
    )


def edit_with_qwen_image_edit(
    image_path: str,
    prompt_text: str,
    generation_settings: Optional[Dict[str, Any]] = None,
) -> str:
    return edit_with_dashscope_image_model(
        image_path,
        prompt_text,
        QWEN_IMAGE_EDIT_MAX_MODEL,
        use_negative_prompt=True,
        generation_settings=generation_settings,
    )


def edit_with_zhipu_image(image_path: str, prompt_text: str) -> str:
    if not ZHIPU_API_KEY:
        raise RuntimeError("未配置 ZHIPU_API_KEY")
    image_data_url = image_file_to_data_url(image_path)
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": ZHIPU_IMAGE_MODEL,
        "prompt": prompt_text,
        "image": image_data_url,
        "size": "1024x1536",
    }
    resp = requests.post(ZHIPU_IMAGE_URL, headers=headers, json=payload, timeout=180)
    if resp.status_code != 200:
        raise requests.HTTPError(response=resp)
    data = resp.json()
    hit = parse_image_url_any(data)
    if not hit:
        raise RuntimeError("智谱图像接口未返回图片 URL")
    return hit


def edit_with_doubao_seedream(
    image_path: str,
    prompt_text: str,
    generation_settings: Optional[Dict[str, Any]] = None,
) -> str:
    if not DOUBAO_API_KEY:
        raise RuntimeError("未配置 DOUBAO_API_KEY")
    image_data_url = image_file_to_data_url(image_path)
    headers = {
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
        "Content-Type": "application/json",
    }
    requested_size = str((generation_settings or {}).get("doubao_size") or "2K")
    model_candidates = build_doubao_model_candidates()
    if not model_candidates:
        raise RuntimeError("豆包可用模型为空，请配置 DOUBAO_IMAGE_MODEL 或 DOUBAO_ENDPOINT_ID")
    normalized = normalize_doubao_url(DOUBAO_IMAGE_URL)
    url_candidates: List[str] = []
    if normalized:
        url_candidates.append(normalized)
    expanded: List[str] = []
    for u in url_candidates:
        expanded.append(u)
        if "/images/generations" in u:
            expanded.append(u.replace("/images/generations", "/images/edits"))
        elif "/images/edits" in u:
            expanded.append(u.replace("/images/edits", "/images/generations"))
    url_candidates = expanded
    seen: set = set()
    ordered_candidates: List[str] = []
    for u in url_candidates:
        if u not in seen:
            ordered_candidates.append(u)
            seen.add(u)
    last_error: Optional[str] = None
    for endpoint in ordered_candidates:
        for model_name in model_candidates:
            # Seedream 5.0 更适合连贯的自然语言，避免堆叠大量英文风格词。
            final_prompt = (
                f"{prompt_text}\n\n"
                "输出为真实摄影照片，保留原图人物身份、环境与主要元素；"
                "不要卡通、动漫、3D 渲染、绘画或插画风格。"
            )

            payload: Dict[str, Any] = {
                "model": model_name,
                "prompt": final_prompt,
            }
            if "/images/generations" in endpoint:
                # 方舟 ImageGenerations 的参考图是字符串数组。
                payload.update({
                    "image": [image_data_url],
                    "size": requested_size,
                    "sequential_image_generation": "disabled",
                    "response_format": "url",
                    "watermark": False,
                })
            else:
                # 兼容用户已配置的旧 /images/edits 接口。
                payload.update({
                    "image": image_data_url,
                    "size": requested_size,
                    "n": 1,
                    "watermark": False,
                })
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=180)
            except requests.exceptions.SSLError as e:
                last_error = f"TLS握手失败：{e}"
                print(f"=== Doubao SSL异常 (可重试): {endpoint} ===")
                # 仅打印简略信息，避免刷屏
                print(f"SSLError: {e}")
                continue
            except Exception as e:
                last_error = str(e)
                print(f"=== Doubao-Seedream-5.0 请求异常: {endpoint} ===")
                print(str(e))
                continue
            print(f"=== Doubao-Seedream-5.0 状态码: {endpoint} | model={model_name} ===")
            print(resp.status_code)
            print(resp.text[:2500])
            if resp.status_code != 200:
                body_text = safe_resp_text(resp)
                if "InvalidEndpointOrModel.NotFound" in body_text:
                    if not str(model_name).lower().startswith("ep-"):
                        last_error = "当前模型无权限或不存在，已自动尝试其他模型；建议将 DOUBAO_IMAGE_MODEL 改为你账号已开通的 ep-xxxx"
                    else:
                        last_error = "豆包 endpoint 不可用或无权限，请确认该 ep-xxxx 已开通图像编辑能力"
                else:
                    last_error = body_text[:400]
                continue
            data = resp.json()
            hit = parse_image_url_any(data)
            if hit:
                return hit
            b64_hit = parse_image_base64_any(data)
            if b64_hit:
                return save_base64_image_to_uploads(b64_hit, "generated_raw")
            last_error = "豆包图像接口未返回图片 URL 或 base64"
    if model_candidates:
        candidate_text = "、".join(model_candidates[:6])
        raise RuntimeError((last_error or "豆包接口调用失败") + f"；已尝试模型：{candidate_text}")
    raise RuntimeError(last_error or "豆包接口调用失败")


def poll_jimeng_result(task_id: str, req_key: str = "", max_rounds: int = 24, sleep_seconds: float = 2.5) -> str:
    if not task_id.strip():
        raise RuntimeError("即梦未返回有效 task_id")
    # 优先使用传入的 req_key，如果为空则尝试配置中的 key
    current_req_key = req_key or JIMENG_REQ_KEY
    # 记录最后一次的错误
    last_error_msg = ""
    
    print(f"=== 开始轮询即梦任务: {task_id} (req_key={current_req_key}) ===")
    
    for i in range(max_rounds):
        time.sleep(sleep_seconds)
        query_body = {
            "req_key": current_req_key,
            "task_id": task_id,
        }
        try:
            # 用户提示 Action=CVGetResult
            result_resp = post_jimeng_signed(
                JIMENG_IMAGE_URL,
                body=query_body,
                action="CVGetResult", 
                version="2022-08-31",
                timeout=90,
            )
            print(f"[Jimeng Poll #{i+1}] Status: {result_resp.status_code}")
            
            if result_resp.status_code != 200:
                last_error_msg = f"HTTP {result_resp.status_code}: {safe_resp_text(result_resp)}"
                continue
                
            data = result_resp.json()
            # 打印状态概要，避免打印超长 Base64
            msg = data.get("message", "")
            status_code_api = data.get("code")
            print(f"[Jimeng Poll Response] Code={status_code_api} Msg={msg}")
            
            hit = parse_image_url_any(data)
            if hit:
                print(f"=== 即梦轮询成功: {hit} ===")
                return hit
            b64_hit = parse_image_base64_any(data)
            if b64_hit:
                print("=== 即梦轮询成功 (Base64) ===")
                return save_base64_image_to_uploads(b64_hit, "jimeng_raw")
                
            payload_text = json.dumps(data, ensure_ascii=False)
            status = data.get("status")
            # 检查显式失败
            if status == "FAIL" or status == "FAILED" or (isinstance(status, int) and status < 0):
                 raise RuntimeError(f"即梦任务返回失败状态: {payload_text[:500]}")
            
            # 如果 resp 有 message 字段且含 error，也可以视作失败
            if "message" in data and "Success" not in data["message"] and "success" not in data["message"]:
                 # 有些接口 processing 时 message 也是 success，需小心
                 pass
                 
        except Exception as e:
            last_error_msg = str(e)
            print(f"[Jimeng Poll Error] {e}")
            continue
            
    raise RuntimeError(f"即梦任务轮询超时或失败，最后错误：{last_error_msg}")


def edit_with_jimeng_image(image_path: str, prompt_text: str) -> str:
    if not JIMENG_IMAGE_URL:
        raise RuntimeError("未配置 JIMENG_IMAGE_URL")
    if not JIMENG_ACCESS_KEY_ID or not JIMENG_SECRET_ACCESS_KEY:
        raise RuntimeError("未配置即梦 AccessKey")
    b64_part = image_file_to_jimeng_base64(image_path)
    req_keys = build_jimeng_req_key_candidates(JIMENG_REQ_KEY)
    last_error = ""
    # 强制风格：写实、非卡通 (使用自然语言否定)
    # 混元 SDK 可能有长度限制，这里做截断处理
    prefix = "photorealistic, real photo, 8k, raw photo, realistic texture. "
    suffix = ". Do not use cartoon style. Do not use anime style. Do not use 3d render style. Do not use painting style."
    
    # 预留长度给 prefix 和 suffix，总长度控制在 950 以内 (混元限制约 1024)
    # prefix(约60) + suffix(约100) = 160，预留 800 给 prompt_text
    max_text_len = 800
    safe_prompt_text = prompt_text
    if len(safe_prompt_text) > max_text_len:
        safe_prompt_text = safe_prompt_text[:max_text_len]
    
    final_prompt = f"{prefix}{safe_prompt_text}{suffix}"
    
    for rk in req_keys:
        payload = {
            "req_key": rk,
            "prompt": final_prompt,
            "binary_data_base64": [b64_part],
            "scale": 0.5,
            "force_single": True,
        }
        resp = post_jimeng_signed(
            JIMENG_IMAGE_URL,
            body=payload,
            action="CVSync2AsyncSubmitTask",
            version="2022-08-31",
            timeout=180,
        )
        print("=== jimeng-ai-image-4.0 提交状态码 ===")
        print(resp.status_code)
        print(resp.text[:2500])
        if resp.status_code != 200:
            last_error = safe_resp_text(resp)[:500]
            continue
        data = resp.json()
        direct_text = json.dumps(data, ensure_ascii=False)
        if any(x in direct_text for x in ["InvalidAccessKeyId", "SignatureNotMatch", "AccessDenied"]):
            raise RuntimeError(f"即梦鉴权失败：{direct_text[:500]}")
        hit = parse_image_url_any(data)
        if hit:
            return hit
        b64_hit = parse_image_base64_any(data)
        if b64_hit:
            return save_base64_image_to_uploads(b64_hit, "jimeng_raw")
        task_id = parse_task_id_any(data)
        if task_id:
            return poll_jimeng_result(task_id, req_key=rk)
        last_error = direct_text[:500]
    raise RuntimeError(f"即梦图像接口未返回图片地址，返回：{last_error or '空响应'}")


def edit_with_hunyuan_image(image_path: str, prompt_text: str) -> str:
    # 重新加载 API Key，确保热重载生效
    global HUNYUAN_API_KEY, HUNYUAN_SECRET_ID, HUNYUAN_SECRET_KEY
    HUNYUAN_API_KEY = load_env("HUNYUAN_API_KEY")
    HUNYUAN_SECRET_ID = load_env("HUNYUAN_SECRET_ID")
    HUNYUAN_SECRET_KEY = load_env("HUNYUAN_SECRET_KEY")
    
    # 强制风格：写实、非卡通
    prefix = "photorealistic, real photo, 8k, raw photo, realistic texture. "
    # 将否定词放入 NegativePrompt，不放在 Prompt 中
    negative_prompt = "cartoon style, anime style, 3d render style, painting style, illustration style, distorted, bad anatomy, wrong perspective, low quality, worst quality, text, watermark, signature, logo"
    
    # 混元 SDK 可能有长度限制，这里做截断处理
    max_text_len = 800
    safe_prompt_text = prompt_text
    
    # 尝试提取 "目标结果描述" 以简化 prompt
    try:
        # 匹配 "目标结果描述：" 到 "理想拍摄结果描述：" 之间的内容
        match = re.search(r"目标结果描述：\s*(.*?)\s*理想拍摄结果描述：", prompt_text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            if extracted and len(extracted) > 10:
                print(f"=== Hunyuan Extracted Prompt: {extracted[:50]}... ({len(extracted)} chars) ===")
                safe_prompt_text = extracted
    except Exception as e:
        print(f"[Hunyuan Prompt Extract Error] {e}")

    if len(safe_prompt_text) > max_text_len:
        safe_prompt_text = safe_prompt_text[:max_text_len]
        
    final_prompt = f"{prefix}{safe_prompt_text}"
    
    # 优先检查 SDK 配置
    if (HUNYUAN_SECRET_ID or (HUNYUAN_API_KEY and HUNYUAN_API_KEY.startswith("AKID"))) and (HUNYUAN_SECRET_KEY):
        if not TENCENT_SDK_AVAILABLE:
            raise RuntimeError("检测到腾讯云 SecretId/Key，但未安装 tencentcloud-sdk-python，无法调用混元 SDK。")
        
        # 使用 SDK 调用
        try:
            sid = HUNYUAN_SECRET_ID or HUNYUAN_API_KEY
            skey = HUNYUAN_SECRET_KEY
            cred = credential.Credential(sid, skey)
            httpProfile = HttpProfile()
            httpProfile.endpoint = "hunyuan.tencentcloudapi.com"
            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile
            # 混元生图通常只支持 ap-guangzhou 或 ap-shanghai
            client = hunyuan_client.HunyuanClient(cred, "ap-guangzhou", clientProfile)
            
            # 读取图片并转 Base64
            with open(image_path, "rb") as f:
                img_data = f.read()
                base64_data = base64.b64encode(img_data).decode("utf-8")
                
            # 使用 SubmitHunyuanImageJob 提交任务 (异步)
            req = hunyuan_models.SubmitHunyuanImageJobRequest()
            params = {
                "Prompt": final_prompt,
                "NegativePrompt": negative_prompt,
                "ContentImage": {
                    "ImageBase64": base64_data
                },
                "Resolution": "1024:1024",
                "Num": 1,
                "StyleId": "201",
                "Revise": 0,  # 关闭提示词扩充
                "LogoAdd": 0  # 尝试关闭水印
            }
            req.from_json_string(json.dumps(params))
            resp = client.SubmitHunyuanImageJob(req)
            job_id = resp.JobId
            print(f"=== Hunyuan Job Submitted: {job_id} ===")
            
            # 轮询结果
            for i in range(30): # Max 60 seconds
                time.sleep(2)
                query_req = hunyuan_models.QueryHunyuanImageJobRequest()
                query_req.JobId = job_id
                query_resp = client.QueryHunyuanImageJob(query_req)
                
                status_code = query_resp.JobStatusCode
                if status_code == "5": # Success
                    print(f"=== Hunyuan Job Success ===")
                    result_images = query_resp.ResultImage
                    if result_images and len(result_images) > 0:
                        img_url = result_images[0]
                        # 下载图片并保存
                        try:
                            img_resp = requests.get(img_url, timeout=30)
                            if img_resp.status_code == 200:
                                b64_res = base64.b64encode(img_resp.content).decode("utf-8")
                                return save_base64_image_to_uploads(b64_res, "hunyuan_raw")
                        except Exception as dl_err:
                            print(f"[Hunyuan Download Error] {dl_err}")
                            # Fallback: return URL directly if download fails (though usually invalid after 1h)
                            return img_url
                            
                elif status_code == "4": # Failed
                    raise RuntimeError(f"Hunyuan Job Failed: {query_resp.JobErrorMsg}")
                
                # Continue if "1" (Wait) or "2" (Run)
                print(f"[Hunyuan Poll #{i+1}] Status: {query_resp.JobStatusMsg}")
            
            raise RuntimeError("Hunyuan Job Polling Timeout")
            
        except Exception as e:
            # 某些 SDK 版本可能没有 GetHunyuanImageToImage，或者参数不同
            # 这里捕获异常并打印详细信息
            error_msg = str(e)
            print(f"[Hunyuan SDK Error] {error_msg}")
            if "InvalidParameter" in error_msg:
                 raise RuntimeError(f"混元 SDK 参数错误: {error_msg}")
            raise RuntimeError(f"混元 SDK 调用失败: {error_msg}")

    # Fallback to OpenAI compatible check
    if not HUNYUAN_API_KEY:
        raise RuntimeError("未配置 HUNYUAN_API_KEY 或 (HUNYUAN_SECRET_ID + HUNYUAN_SECRET_KEY)")
    
    # 检查 Key 格式
    if HUNYUAN_API_KEY.startswith("AKID"):
        if not HUNYUAN_SECRET_KEY:
             raise RuntimeError("检测到 HUNYUAN_API_KEY 为 SecretId (以 AKID 开头)，但未配置 HUNYUAN_SECRET_KEY，无法进行签名认证。")
        # 理论上上面 SDK 逻辑应该覆盖了，如果走到这里说明 SDK import 失败或者其他逻辑错误
        raise RuntimeError("检测到 SecretId/Key 但 SDK 调用未生效，请检查环境。")
        
    # 尝试调用 OpenAI 兼容接口 (假设存在)
    # 注意: 混元目前主要通过 SDK 提供图像能力，OpenAI 接口可能仅限对话
    # 这里做一个尝试性的请求，如果失败则抛出异常
    
    # 构造请求 (假设路径)
    url = "https://api.hunyuan.cloud.tencent.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {HUNYUAN_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "hunyuan-dit",
        "prompt": final_prompt,
        "n": 1,
        "size": "1024x1024"
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
             data = resp.json()
             hit = parse_image_url_any(data)
             if hit: return hit
             b64 = parse_image_base64_any(data)
             if b64: return save_base64_image_to_uploads(b64, "hunyuan_raw")
    except Exception:
        pass

    raise RuntimeError("混元图像生成接口调用失败或暂不支持。请确认 API Key 权限或等待 SDK 集成。")


def get_image_model_runner(model_name: str):
    return {
        MODEL_WAN_27: edit_with_wan27_image,
        MODEL_QWEN_IMAGE_20: edit_with_qwen_image_20,
        MODEL_SEEDREAM_50: edit_with_doubao_seedream,
        MODEL_QWEN_EDIT_MAX: edit_with_qwen_image_edit,
    }.get(model_name)


def check_image_model_ready(model_name: str) -> Tuple[bool, str]:
    if model_name in {MODEL_WAN_27, MODEL_QWEN_IMAGE_20, MODEL_QWEN_EDIT_MAX}:
        return bool(DASHSCOPE_API_KEY), "未配置 DASHSCOPE_API_KEY"
    if model_name == MODEL_SEEDREAM_50:
        return bool(DOUBAO_API_KEY), "未配置 DOUBAO_API_KEY"
    if model_name == "zhipu-glm-image":
        return bool(ZHIPU_API_KEY), "未配置 ZHIPU_API_KEY"
    if model_name == "jimeng-ai-image-4.0":
        if not JIMENG_IMAGE_URL:
            return False, "未配置 JIMENG_IMAGE_URL"
        if not JIMENG_ACCESS_KEY_ID or not JIMENG_SECRET_ACCESS_KEY:
            return False, "未配置即梦 AccessKey"
        return True, ""
    if model_name == "hunyuan-dit":
        return bool(HUNYUAN_API_KEY) or (bool(HUNYUAN_SECRET_ID) and bool(HUNYUAN_SECRET_KEY)), "未配置 HUNYUAN_API_KEY 或 SecretId/Key"
    return False, f"未知模型：{model_name}"


def get_ready_models(model_names: List[str]) -> List[str]:
    ready: List[str] = []
    for model_name in model_names:
        ok, _ = check_image_model_ready(model_name)
        if ok:
            ready.append(model_name)
    return ready


def run_model_compare(
    image_path: str,
    prompt_text: str,
    generation_settings: Optional[Dict[str, Any]] = None,
    user_intent: Optional[Dict[str, str]] = None,
    on_candidate: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """并发调用多个生图模型，谁先成功谁作为主图（不用等全部跑完再选优）。

    返回结构保持兼容：diagram_url/image_model 为主图结果，
    model_candidates 为全部候选（按 MODEL_COMPARE_LIST 顺序），供前端对比展示。

    on_candidate: 每完成一个模型（成功或失败）就调用一次 (model_name, item)，
    用于增量通知（例如把已完成的候选实时写回 latest.json，前端逐张显示）。
    """
    active_models = get_ready_models(MODEL_COMPARE_LIST)
    if not active_models:
        return {
            "diagram_url": "",
            "raw_diagram_url": "",
            "image_model": "",
            "diagram_error": "未配置任何可用图像模型。",
            "beauty_applied": False,
            "model_candidates": [],
        }

    results: Dict[str, Dict[str, Any]] = {}
    first_ok_model: List[str] = []
    lock = threading.Lock()
    first_ok_lock = threading.Lock()

    def _valid_url(url: str) -> bool:
        s = str(url).strip()
        return bool(s) and (
            s.startswith("http://")
            or s.startswith("https://")
            or s.startswith("/uploads/")
        )

    def _worker(model_name: str) -> None:
        runner = get_image_model_runner(model_name)
        if runner is None:
            return
        try:
            raw_url = runner(image_path, prompt_text, generation_settings)
            if not _valid_url(raw_url):
                raise RuntimeError("模型返回空图片地址。")
            # 第一个成功者即标记为主图模型，不阻塞等待质量打分
            with first_ok_lock:
                if not first_ok_model:
                    first_ok_model.append(model_name)
            quality_score, quality_reason = evaluate_image_quality(
                model_name,
                raw_url,
                generation_settings=generation_settings,
                user_intent=user_intent,
            )
            item = {
                "model": model_name,
                "raw_diagram_url": raw_url,
                "diagram_url": raw_url,
                "beauty_applied": False,
                "error": "",
                "quality_score": quality_score,
                "quality_reason": quality_reason,
            }
        except Exception as e:
            err = str(e)
            if not err and isinstance(e, requests.HTTPError):
                try:
                    err = safe_resp_text(e.response)
                except Exception:
                    err = ""
            if "InternalError" in err or "submit algo service error" in err:
                err = "服务繁忙（InternalError），已自动回退"
            item = {
                "model": model_name,
                "raw_diagram_url": "",
                "diagram_url": "",
                "beauty_applied": False,
                "error": err,
                "quality_score": -999.0,
                "quality_reason": "",
            }
        with lock:
            results[model_name] = item
        if on_candidate is not None:
            try:
                on_candidate(model_name, item)
            except Exception:
                pass

    threads = [
        threading.Thread(target=_worker, args=(m,), daemon=True)
        for m in active_models
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    candidates = [results[m] for m in active_models if m in results]
    chosen_model = first_ok_model[0] if first_ok_model else ""
    chosen = results.get(chosen_model) if chosen_model else None

    # 若首个成功模型因质量检测被标记失败（极少见），回退到任一成功候选
    if chosen is None or chosen.get("error"):
        chosen = next((c for c in candidates if not c.get("error")), None)

    if chosen:
        return {
            "diagram_url": chosen["diagram_url"],
            "raw_diagram_url": chosen["raw_diagram_url"],
            "image_model": chosen["model"],
            "diagram_error": "",
            "beauty_applied": chosen["beauty_applied"],
            "model_candidates": candidates,
        }

    errors = [c.get("error", "") for c in candidates if c.get("error")]
    return {
        "diagram_url": "",
        "raw_diagram_url": "",
        "image_model": "",
        "diagram_error": "；".join(errors) or "所有生图模型调用失败。",
        "beauty_applied": False,
        "model_candidates": candidates,
    }


def edit_image_auto(
    image_path: str,
    prompt_text: str,
    generation_settings: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, str]:
    errors: List[str] = []
    for model_name in get_ready_models(IMAGE_EDIT_MODELS):
        runner = get_image_model_runner(model_name)
        if runner is None:
            continue
        try:
            return runner(image_path, prompt_text, generation_settings), model_name, ""
        except Exception as e:
            err = f"{model_name} 失败：{e}"
            errors.append(err)
            print("[图像编辑失败]", err)
    if not errors:
        return "", "", "未配置任何可用图像模型，请先配置 DASHSCOPE_API_KEY 或 DOUBAO_API_KEY。"
    return "", "", "；".join(errors)


# =========================
# 路由
# =========================
@app.route("/")
def root():
    return render_template(
        "mobile.html",
        optimization_schema=OPTIMIZATION_SCHEMA,
        default_user_intent=DEFAULT_USER_INTENT,
    )


@app.route("/mobile")
def mobile_page():
    return render_template(
        "mobile.html",
        optimization_schema=OPTIMIZATION_SCHEMA,
        default_user_intent=DEFAULT_USER_INTENT,
    )


@app.route("/display")
def display_page():
    return render_template("display.html")


@app.route("/qrcode")
def qrcode_page():
    host_ip = get_local_ip()
    qr_info = generate_access_qrcodes(host_ip, port=5000)
    return render_template("qrcode.html", qr_info=qr_info)


@app.route("/api/esp_config", methods=["GET", "POST"])
def api_esp_config():
    try:
        if request.method == "GET":
            cfg = get_runtime_esp_config()
            return jsonify({
                "ok": True,
                "esp_enabled": cfg["enabled"],
                "esp_ip": cfg["ip"],
                "esp_port": cfg["port"],
                "esp_endpoint": cfg["endpoint"],
                "esp_size": cfg["size"],
            })

        payload = request.get_json(silent=True) or {}
        updates: Dict[str, str] = {}
        if "esp_enabled" in payload:
            raw_enabled = payload.get("esp_enabled")
            enabled = raw_enabled if isinstance(raw_enabled, bool) else parse_bool(str(raw_enabled), True)
            updates["ESP_SCREEN_ENABLED"] = "1" if enabled else "0"
        esp_ip = str(payload.get("esp_ip", "")).strip()
        if esp_ip:
            if not is_valid_ipv4(esp_ip):
                return jsonify({"ok": False, "msg": "ESP IP 格式不正确，请填写 IPv4 地址"}), 400
            updates["ESP_SCREEN_IP"] = esp_ip
        if not updates:
            return jsonify({"ok": False, "msg": "没有可保存的 ESP 配置"}), 400

        update_api_keys_file(updates)
        cfg = get_runtime_esp_config()
        msg_parts = []
        if "ESP_SCREEN_IP" in updates:
            msg_parts.append(f"屏幕 IP：{updates['ESP_SCREEN_IP']}")
        if "ESP_SCREEN_ENABLED" in updates:
            msg_parts.append("小屏幕已启用" if updates["ESP_SCREEN_ENABLED"] == "1" else "小屏幕已关闭")
        saved_msg = "已保存。" + (" " + "；".join(msg_parts) if msg_parts else "")
        return jsonify({
            "ok": True,
            "msg": saved_msg,
            "esp_enabled": cfg["enabled"],
            "esp_ip": cfg["ip"],
            "esp_port": cfg["port"],
            "esp_endpoint": cfg["endpoint"],
            "esp_size": cfg["size"],
        })
    except Exception as e:
        return jsonify({"ok": False, "msg": f"保存 ESP 配置失败：{e}"}), 500


@app.route("/api/diagnose", methods=["POST"])
def api_diagnose():
    try:
        if "photo" not in request.files:
            return jsonify({"ok": False, "msg": "没有收到图片字段 photo"}), 400

        file = request.files["photo"]
        if file.filename == "":
            return jsonify({"ok": False, "msg": "未选择图片"}), 400

        new_filename = build_upload_filename(file.filename)

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], new_filename)
        file.save(save_path)

        print(f"[已接收] {save_path}")
        print("[开始基础摄影诊断]")

        geo = extract_geometry(save_path)
        geometry_block = geo["prompt_block"] if geo else ""
        ai_result, used_vision_model = call_vision_auto(save_path, geometry_block)
        preview_url = f"/uploads/{new_filename}"
        # 诊断阶段只保存原图，不推送到屏幕；屏幕仅展示 AI 生成结果。
        esp_push_ok, esp_push_msg = False, "等待 AI 生成后再推送到 ESP"
        print(f"[ESP] {esp_push_msg}")

        session_id = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:6]
        session_payload = {
            "session_id": session_id,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": new_filename,
            "preview_url": preview_url,
            "image_path": save_path,
            "vision_model": used_vision_model,
            "detector_data": geo,
            "stage": "diagnosed",
            "diagnosis_report": ai_result,
            "user_intent": None,
            "strategy_plan": None,
            "edit_prompt": "",
            "raw_diagram_url": "",
            "diagram_url": "",
            "diagram_error": "",
            "image_model": "",
            "beauty_applied": False,
            "model_candidates": [],
            "esp_push_ok": esp_push_ok,
            "esp_push_msg": esp_push_msg,
            "esp_pushed_image": "",
            "esp_push_count": 0,
            "esp_push_total": 0,
        }
        save_session_data(session_id, session_payload)

        latest_payload = {
            "ok": True,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "stage": "diagnosed",
            "filename": new_filename,
            "preview_url": preview_url,
            "vision_model": used_vision_model,
            "detector_data": geo,
            "ai_result": ai_result,
            "diagnosis_report": ai_result,
            "strategy_plan": None,
            "user_intent": None,
            "edit_prompt": "",
            "raw_diagram_url": "",
            "diagram_url": "",
            "diagram_error": "",
            "image_model": "",
            "beauty_applied": False,
            "model_candidates": [],
            "esp_push_ok": esp_push_ok,
            "esp_push_msg": esp_push_msg,
            "esp_pushed_image": "",
            "esp_push_count": 0,
            "esp_push_total": 0,
        }
        save_latest_result(latest_payload)

        return jsonify({
            "ok": True,
            "msg": "拍照上传成功，已完成基础摄影诊断。",
            "session_id": session_id,
            "stage": "diagnosed",
            "filename": new_filename,
            "preview_url": preview_url,
            "vision_model": used_vision_model,
            "ai_result": ai_result,
            "diagnosis_report": ai_result,
            "esp_push_ok": esp_push_ok,
            "esp_push_msg": esp_push_msg,
            "esp_pushed_image": "",
            "esp_push_count": 0,
            "esp_push_total": 0,
        })

    except ValueError as e:
        return jsonify({"ok": False, "msg": str(e)}), 400

    except requests.HTTPError as e:
        err_text = ""
        try:
            err_text = e.response.text
        except Exception:
            err_text = str(e)

        latest_payload = {
            "ok": False,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": f"接口 HTTP 错误：{err_text}"
        }
        save_latest_result(latest_payload)
        return jsonify({"ok": False, "msg": "模型接口调用失败，请查看电脑端日志。"}), 500

    except Exception as e:
        latest_payload = {
            "ok": False,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": f"处理失败：{str(e)}"
        }
        save_latest_result(latest_payload)
        return jsonify({"ok": False, "msg": f"处理失败：{str(e)}"}), 500


@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        payload = request.get_json(silent=True) or {}
        if payload.get("generation_confirmed") is not True:
            return jsonify({
                "ok": False,
                "msg": "未收到生成按钮的明确确认，已拒绝启动图像生成。",
            }), 400

        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            return jsonify({"ok": False, "msg": "缺少 session_id"}), 400
        try:
            session_id = validate_session_id(session_id)
        except ValueError as e:
            return jsonify({"ok": False, "msg": str(e)}), 400

        session_data = load_session_data(session_id)
        image_path = session_data.get("image_path", "")
        if not image_path or not os.path.exists(image_path):
            return jsonify({"ok": False, "msg": "会话原图不存在，请重新拍照。"}), 400

        diagnosis_report = session_data.get("diagnosis_report", {})
        if not isinstance(diagnosis_report, dict):
            diagnosis_report = {}

        user_intent = normalize_user_intent(payload.get("user_intent", {}))
        generation_settings = build_generation_settings(user_intent)
        debug_compare = parse_bool(
            str(payload.get("debug_compare", MODEL_COMPARE_ENABLED_DEFAULT)),
            MODEL_COMPARE_ENABLED_DEFAULT,
        )
        strategy_plan = build_strategy_plan(diagnosis_report, user_intent)
        targeted_ideal_prompt = build_targeted_ideal_prompt(diagnosis_report, strategy_plan, user_intent)
        edit_prompt = build_edit_prompt(diagnosis_report, strategy_plan, user_intent, targeted_ideal_prompt)
        final_guidance = build_guidance_text(diagnosis_report, strategy_plan, user_intent)
        final_guidance["ideal_image_prompt"] = targeted_ideal_prompt

        print("[开始图像编辑生成理想拍摄结果图]")
        model_candidates: List[Dict[str, Any]] = []
        if debug_compare:
            # 生成中：先写 "generating" 快照，让 display 页立即展示诊断信息
            generating_payload = {
                "ok": True,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": session_id,
                "stage": "generating",
                "filename": session_data.get("filename", ""),
                "preview_url": session_data.get("preview_url", ""),
                "vision_model": session_data.get("vision_model", ""),
                "image_model": "",
                "diagnosis_report": diagnosis_report,
                "user_intent": user_intent,
                "generation_settings": generation_settings,
                "strategy_plan": strategy_plan,
                "ai_result": final_guidance,
                "edit_prompt": edit_prompt,
                "raw_diagram_url": "",
                "diagram_url": "",
                "diagram_error": "",
                "beauty_applied": False,
                "beauty_enabled": False,
                "model_candidates": [],
                "esp_push_ok": False,
                "esp_push_msg": "",
                "esp_pushed_image": "",
                "esp_push_count": 0,
                "esp_push_total": 0,
            }
            save_latest_result(generating_payload)

            # 每完成一个模型，就把已完成候选增量写回，前端逐张显示，不用等全部完成
            partial_results: Dict[str, Dict[str, Any]] = {}

            def on_candidate(model_name: str, item: Dict[str, Any]) -> None:
                partial_results[model_name] = item
                ordered = [partial_results[m] for m in MODEL_COMPARE_LIST if m in partial_results]
                snapshot = dict(generating_payload)
                snapshot["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                snapshot["model_candidates"] = ordered
                first_ok = next((c for c in ordered if not c.get("error")), None)
                if first_ok:
                    snapshot["diagram_url"] = first_ok.get("diagram_url", "")
                    snapshot["image_model"] = first_ok.get("model", "")
                save_latest_result(snapshot)

            compare_data = run_model_compare(
                image_path,
                edit_prompt,
                generation_settings=generation_settings,
                user_intent=user_intent,
                on_candidate=on_candidate,
            )
            diagram_url = compare_data.get("diagram_url", "")
            used_image_model = compare_data.get("image_model", "")
            diagram_error = compare_data.get("diagram_error", "")
            raw_diagram_url = compare_data.get("raw_diagram_url", diagram_url)
            beauty_applied = bool(compare_data.get("beauty_applied", False))
            model_candidates = compare_data.get("model_candidates", [])
        else:
            diagram_url, used_image_model, diagram_error = edit_image_auto(
                image_path,
                edit_prompt,
                generation_settings=generation_settings,
            )
            raw_diagram_url = diagram_url
            beauty_applied = False
            model_candidates = [{
                "model": used_image_model,
                "raw_diagram_url": raw_diagram_url,
                "diagram_url": diagram_url,
                "beauty_applied": beauty_applied,
                "error": diagram_error if not diagram_url else "",
            }]

        # 仅推送 AI 生成图，不再回退推送原始拍照图。
        esp_target_image = str(diagram_url or "").strip()
        # 自动流程推送多图队列（主图 + 其余候选图），供屏幕端滑动查看。
        esp_queue = build_esp_push_queue(esp_target_image, model_candidates, include_all=True)
        if not esp_queue:
            esp_push_ok, esp_push_msg = False, "No image to push."
            esp_push_count = 0
            esp_push_total = 0
        else:
            success_count = 0
            details: List[str] = []
            for i, ref in enumerate(esp_queue):
                ok, msg = push_image_to_esp(ref, append=i > 0)
                details.append(msg)
                if ok:
                    success_count += 1
            esp_push_ok = success_count > 0
            esp_push_count = success_count
            esp_push_total = len(esp_queue)
            tail = details[-1] if details else ""
            esp_push_msg = f"ESP push {success_count}/{len(esp_queue)}. {tail}"
        print(f"[ESP] {esp_push_msg}")

        session_data["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_data["stage"] = "generated"
        session_data["user_intent"] = user_intent
        session_data["generation_settings"] = generation_settings
        session_data["strategy_plan"] = strategy_plan
        session_data["edit_prompt"] = edit_prompt
        session_data["raw_diagram_url"] = raw_diagram_url
        session_data["diagram_url"] = diagram_url
        session_data["diagram_error"] = diagram_error
        session_data["image_model"] = used_image_model
        session_data["beauty_applied"] = beauty_applied
        session_data["beauty_enabled"] = False
        session_data["model_candidates"] = model_candidates
        session_data["ai_result"] = final_guidance
        session_data["esp_push_ok"] = esp_push_ok
        session_data["esp_push_msg"] = esp_push_msg
        session_data["esp_pushed_image"] = esp_target_image if esp_push_ok else ""
        session_data["esp_push_count"] = esp_push_count
        session_data["esp_push_total"] = esp_push_total
        save_session_data(session_id, session_data)

        latest_payload = {
            "ok": True,
            "time": session_data["time"],
            "session_id": session_id,
            "stage": "generated",
            "filename": session_data.get("filename", ""),
            "preview_url": session_data.get("preview_url", ""),
            "vision_model": session_data.get("vision_model", ""),
            "image_model": used_image_model,
            "diagnosis_report": diagnosis_report,
            "user_intent": user_intent,
            "generation_settings": generation_settings,
            "strategy_plan": strategy_plan,
            "ai_result": final_guidance,
            "edit_prompt": edit_prompt,
            "raw_diagram_url": raw_diagram_url,
            "diagram_url": diagram_url,
            "diagram_error": diagram_error,
            "beauty_applied": beauty_applied,
            "beauty_enabled": False,
            "model_candidates": model_candidates,
            "esp_push_ok": esp_push_ok,
            "esp_push_msg": esp_push_msg,
            "esp_pushed_image": esp_target_image if esp_push_ok else "",
            "esp_push_count": esp_push_count,
            "esp_push_total": esp_push_total,
        }
        save_latest_result(latest_payload)

        return jsonify({
            "ok": True,
            "msg": "已根据你的目标生成专业策略和理想结果图。",
            "session_id": session_id,
            "stage": "generated",
            "preview_url": session_data.get("preview_url", ""),
            "vision_model": session_data.get("vision_model", ""),
            "image_model": used_image_model,
            "diagnosis_report": diagnosis_report,
            "user_intent": user_intent,
            "generation_settings": generation_settings,
            "strategy_plan": strategy_plan,
            "ai_result": final_guidance,
            "raw_diagram_url": raw_diagram_url,
            "diagram_url": diagram_url,
            "diagram_error": diagram_error,
            "beauty_applied": beauty_applied,
            "beauty_enabled": False,
            "model_candidates": model_candidates,
            "esp_push_ok": esp_push_ok,
            "esp_push_msg": esp_push_msg,
            "esp_pushed_image": esp_target_image if esp_push_ok else "",
            "esp_push_count": esp_push_count,
            "esp_push_total": esp_push_total,
        })

    except FileNotFoundError as e:
        return jsonify({"ok": False, "msg": str(e)}), 404

    except requests.HTTPError as e:
        err_text = ""
        try:
            err_text = e.response.text
        except Exception:
            err_text = str(e)
        return jsonify({"ok": False, "msg": f"模型接口调用失败：{err_text}"}), 500

    except Exception as e:
        return jsonify({"ok": False, "msg": f"处理失败：{str(e)}"}), 500


@app.route("/upload", methods=["POST"])
def upload():
    return api_diagnose()


@app.route("/api/push_latest_to_esp", methods=["POST"])
def api_push_latest_to_esp():
    try:
        payload = request.get_json(silent=True) or {}
        image_ref = str(payload.get("image_ref", "")).strip()
        include_all = parse_bool(str(payload.get("include_all", "0")), False)

        latest_data: Dict[str, Any] = {}
        if os.path.exists(LATEST_RESULT_JSON):
            with open(LATEST_RESULT_JSON, "r", encoding="utf-8") as f:
                latest_data = json.load(f)

        allowed_refs = {
            str(latest_data.get("diagram_url", "")).strip(),
            str(latest_data.get("raw_diagram_url", "")).strip(),
        }
        latest_candidates = latest_data.get("model_candidates", [])
        if isinstance(latest_candidates, list):
            for item in latest_candidates:
                if isinstance(item, dict):
                    allowed_refs.add(str(item.get("diagram_url", "")).strip())
                    allowed_refs.add(str(item.get("raw_diagram_url", "")).strip())
        allowed_refs.discard("")

        if image_ref and image_ref not in allowed_refs:
            return jsonify({"ok": False, "msg": "拒绝推送非当前生成结果的图片。"}), 400

        if not image_ref:
            image_ref = str(latest_data.get("diagram_url", "")).strip()

        if not image_ref:
            return jsonify({"ok": False, "msg": "暂无可推送 AI 生成图，请先完成生成。"}), 400

        candidates = latest_data.get("model_candidates", []) if include_all else []
        if not isinstance(candidates, list):
            candidates = []
        queue = build_esp_push_queue(image_ref, candidates, include_all=include_all)
        if not queue:
            return jsonify({"ok": False, "msg": "暂无可推送 AI 生成图，请先完成生成。"}), 400

        success_count = 0
        details: List[str] = []
        for i, ref in enumerate(queue):
            ok, msg = push_image_to_esp(ref, append=i > 0)
            details.append(msg)
            if ok:
                success_count += 1
        ok = success_count > 0
        msg = f"ESP push {success_count}/{len(queue)}. {(details[-1] if details else '')}".strip()

        if latest_data:
            selected_candidate = None
            if isinstance(latest_candidates, list):
                selected_candidate = next(
                    (
                        item for item in latest_candidates
                        if isinstance(item, dict)
                        and str(item.get("diagram_url", "")).strip() == image_ref
                    ),
                    None,
                )
            if ok:
                latest_data["diagram_url"] = image_ref
                if isinstance(selected_candidate, dict):
                    latest_data["raw_diagram_url"] = selected_candidate.get("raw_diagram_url", image_ref)
                    latest_data["image_model"] = selected_candidate.get("model", latest_data.get("image_model", ""))
            latest_data["esp_push_ok"] = ok
            latest_data["esp_push_msg"] = msg
            latest_data["esp_pushed_image"] = image_ref if ok else ""
            latest_data["esp_push_count"] = success_count
            latest_data["esp_push_total"] = len(queue)
            save_latest_result(latest_data)

            session_id = str(latest_data.get("session_id", "")).strip()
            if ok and session_id:
                try:
                    session_data = load_session_data(session_id)
                    session_data["diagram_url"] = latest_data.get("diagram_url", image_ref)
                    session_data["raw_diagram_url"] = latest_data.get("raw_diagram_url", image_ref)
                    session_data["image_model"] = latest_data.get("image_model", "")
                    session_data["esp_pushed_image"] = image_ref
                    save_session_data(session_id, session_data)
                except Exception as e:
                    print(f"[ESP] 更新会话主图失败：{e}")

        return jsonify({
            "ok": ok,
            "msg": msg,
            "image_ref": image_ref,
            "push_count": success_count,
            "push_total": len(queue),
        }), 200 if ok else 500
    except Exception as e:
        return jsonify({"ok": False, "msg": f"ESP 推送失败：{e}"}), 500


@app.route("/api/latest_result")
def api_latest_result():
    if not os.path.exists(LATEST_RESULT_JSON):
        return jsonify({
            "ok": False,
            "msg": "暂无数据，请先用手机拍照上传。"
        })

    with open(LATEST_RESULT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "raw_diagram_url" not in data:
        data["raw_diagram_url"] = data.get("diagram_url", "")
    if "beauty_applied" not in data:
        data["beauty_applied"] = False
    if "beauty_enabled" not in data:
        data["beauty_enabled"] = False
    if "model_candidates" not in data:
        data["model_candidates"] = []
    if "esp_push_ok" not in data:
        data["esp_push_ok"] = False
    if "esp_push_msg" not in data:
        data["esp_push_msg"] = ""
    if "esp_pushed_image" not in data:
        data["esp_pushed_image"] = ""
    if "esp_push_count" not in data:
        data["esp_push_count"] = 0
    if "esp_push_total" not in data:
        data["esp_push_total"] = 0
    return jsonify(data)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    host_ip = get_local_ip()
    qr_info = generate_access_qrcodes(host_ip, port=5000)
    qrcode_url = f"http://127.0.0.1:5000/qrcode"
    display_url = f"http://127.0.0.1:5000/display"

    print("====================================")
    print("服务已准备启动")
    print(f"手机拍照页：{qr_info['mobile_url']}")
    print(f"电脑显示页：{display_url}")
    print(f"二维码页：{qrcode_url}")
    print(f"视觉理解模型顺序：{VISION_MODELS}")
    print(f"图像编辑模型顺序：{get_ready_models(IMAGE_EDIT_MODELS)}")
    print(f"二维码图片已生成：{os.path.join(STATIC_FOLDER, 'qr_mobile.png')}")
    print("====================================")

    # 启动后自动打开二维码页和电脑显示页
    open_browser_later(qrcode_url, delay=1.2)
    open_browser_later(display_url, delay=1.8)

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
