"""配置加载与运行时配置（load_env/get_runtime_esp_config/常量）。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；唯一改写 = D8（BASE_DIR 上溯一级，见阶段①报告）。
from typing import Any, Callable, Dict, List, Optional, Tuple
import ipaddress
import json
import os
import subprocess


# =========================
# 基础配置
# =========================
# 所有本地路径都锚定主程序目录，避免从上级目录或快捷方式启动时，
# 上传图、会话和配置文件被写到意外位置。
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # D8: 拆包后上溯一级回到 P.A
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")
LATEST_RESULT_JSON = os.path.join(STATIC_FOLDER, "latest_result.json")
SESSION_FOLDER = os.path.join(STATIC_FOLDER, "sessions")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_CONTENT_LENGTH = 15 * 1024 * 1024  # 15MB
API_KEYS_FILE = os.path.join(BASE_DIR, "_config", "api_keys.json")


# =========================
# 配置读取（阶段②：进程内缓存 + 文件 mtime 失效）
# =========================
# 优先级逐字保持不变：_config/api_keys.json > 进程环境变量 > Windows 用户环境变量 > 代码默认值。
# 变化点：读盘与 powershell 子进程从"每次取值"移到"首次取值 / 文件变化时"，
# 热路径上只剩 os.stat + dict 查找。
_KEYS_CACHE: Dict[str, Any] = {"stamp": None, "value": {}}
_WIN_ENV_CACHE: Dict[str, Any] = {"stamp": None, "value": {}}


def _api_keys_stamp() -> Tuple[str, Any, Any]:
    """(路径, mtime_ns, size)：三者任一变化即视为需要重读。"""
    try:
        st = os.stat(API_KEYS_FILE)
        return (API_KEYS_FILE, st.st_mtime_ns, st.st_size)
    except OSError:
        return (API_KEYS_FILE, None, None)


def _read_api_keys_file_uncached() -> Dict[str, str]:
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


def load_api_keys_file() -> Dict[str, str]:
    """api_keys.json 内容；文件未变化时零读盘（原实现每次调用都 open+json）。"""
    stamp = _api_keys_stamp()
    if _KEYS_CACHE["stamp"] != stamp:
        _KEYS_CACHE["value"] = _read_api_keys_file_uncached()
        _KEYS_CACHE["stamp"] = stamp
    return dict(_KEYS_CACHE["value"])


def invalidate_api_keys_cache() -> None:
    """让文件内容与 Windows 用户环境变量一起失效（两者共用同一失效信号）。"""
    _KEYS_CACHE["stamp"] = None
    _WIN_ENV_CACHE["stamp"] = None


def _windows_user_env() -> Dict[str, str]:
    """Windows 用户环境变量：与 api_keys.json 共用同一个失效信号。

    原实现按名字逐个 spawn powershell（缺 36 个名字就起 36 次、每次最坏 2.5s）。
    这里改为批量取回一次并缓存；缓存键就是 api_keys.json 的 (mtime,size)，
    所以"改了配置文件"→ 文件缓存与用户环境变量同时刷新，
    "改配置不用重启"这条现有体验对两级都成立，且热路径上零子进程。
    powershell 不可用（非 Windows）时静默降级为空表，与改前一致。
    """
    stamp = _KEYS_CACHE["stamp"]
    if stamp is None:
        # 未经 load_api_keys_file() 预热时自取（直接调用本函数的场景）
        stamp = _api_keys_stamp()
    if _WIN_ENV_CACHE["stamp"] == stamp:
        return _WIN_ENV_CACHE["value"]

    fetched: Dict[str, str] = {}
    try:
        cmd = "[System.Environment]::GetEnvironmentVariables('User') | ConvertTo-Json -Compress"
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=2.5
        )
        data = json.loads((out.stdout or "").strip() or "{}")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str):
                    fetched[str(k)] = v.strip()
    except Exception:
        pass

    # 整体替换引用而非原地改写：GIL 下天然原子，多线程读取不会看到半成品。
    _WIN_ENV_CACHE["value"] = fetched
    _WIN_ENV_CACHE["stamp"] = stamp
    return fetched


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
    # 立即失效缓存：POST /api/esp_config 写完马上要读到新值，
    # 不依赖 mtime 精度（drvfs 上时间戳粒度较粗）。
    invalidate_api_keys_cache()


def load_env(name: str, default: str = "") -> str:
    # 阶段②：文件内容与 Windows 用户环境变量都已进程内缓存，
    # 这一层只剩"缓存命中判断 + 字典查找"，四级优先级顺序与改前逐字一致。
    file_val = str(load_api_keys_file().get(name, "")).strip()
    if file_val:
        return file_val
    val = os.getenv(name, "").strip()
    if val:
        return val
    win_user_val = str(_windows_user_env().get(name, "")).strip()
    if win_user_val:
        return win_user_val
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
ZHIPU_VISION_MODEL = load_env("ZHIPU_VISION_MODEL", "glm-5.3-flash")
# 思考力度（文档：thinking.type 仅支持 enabled，力度走顶层 reasoning_effort low/high/max）
# low = 零思考，实测 ~25-36s、质量与默认档（reasoning ~2000 tokens、30-120s）无肉眼差异
ZHIPU_REASONING_EFFORT = load_env("ZHIPU_REASONING_EFFORT", "low")

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
HUNYUAN_CHAT_URL = load_env("HUNYUAN_CHAT_URL", "https://tokenhub.tencentmaas.com/v1/chat/completions")
HUNYUAN_VISION_MODEL = load_env("HUNYUAN_VISION_MODEL", "hy-vision-2.0-instruct")

# DeepSeek 视觉理解配置
DEEPSEEK_API_KEY = load_env("DEEPSEEK_API_KEY")
DEEPSEEK_CHAT_URL = load_env("DEEPSEEK_CHAT_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_VISION_MODEL = load_env("DEEPSEEK_VISION_MODEL", "deepseek-flash")

# ESP32 屏幕推送配置
ESP_SCREEN_ENABLED = parse_bool(load_env("ESP_SCREEN_ENABLED", "1"), True)
ESP_SCREEN_IP = load_env("ESP_SCREEN_IP")
ESP_SCREEN_PORT = parse_int(load_env("ESP_SCREEN_PORT", "80"), 80)
ESP_SCREEN_ENDPOINT = load_env("ESP_SCREEN_ENDPOINT", "/img")
ESP_SCREEN_SIZE = load_env("ESP_SCREEN_SIZE", "320x240")
ESP_SCREEN_JPEG_QUALITY = parse_int(load_env("ESP_SCREEN_JPEG_QUALITY", "85"), 85)
# 局域网内 3 秒足够：屏幕不通时让学生等 6 秒只会显得程序卡住
ESP_SCREEN_TIMEOUT_SEC = parse_float(load_env("ESP_SCREEN_TIMEOUT_SEC", "3"), 3.0)
