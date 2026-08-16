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
from typing import Any, Dict, List, Optional, Tuple
import threading
import webbrowser
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
import requests
import qrcode
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

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
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
LATEST_RESULT_JSON = os.path.join(STATIC_FOLDER, "latest_result.json")
SESSION_FOLDER = os.path.join(STATIC_FOLDER, "sessions")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_CONTENT_LENGTH = 15 * 1024 * 1024  # 15MB
API_KEYS_FILE = os.path.join(os.path.dirname(__file__), "api_keys.json")


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
HUNYUAN_CHAT_URL = load_env("HUNYUAN_CHAT_URL", "https://api.hunyuan.cloud.tencent.com/v1/chat/completions")
HUNYUAN_VISION_MODEL = load_env("HUNYUAN_VISION_MODEL", "hunyuan-vision")

# ESP32 屏幕推送配置
ESP_SCREEN_ENABLED = parse_bool(load_env("ESP_SCREEN_ENABLED", "1"), True)
ESP_SCREEN_IP = load_env("ESP_SCREEN_IP")
ESP_SCREEN_PORT = parse_int(load_env("ESP_SCREEN_PORT", "80"), 80)
ESP_SCREEN_ENDPOINT = load_env("ESP_SCREEN_ENDPOINT", "/img")
ESP_SCREEN_SIZE = load_env("ESP_SCREEN_SIZE", "320x240")
ESP_SCREEN_JPEG_QUALITY = parse_int(load_env("ESP_SCREEN_JPEG_QUALITY", "85"), 85)
ESP_SCREEN_TIMEOUT_SEC = parse_float(load_env("ESP_SCREEN_TIMEOUT_SEC", "6"), 6.0)

# 视觉理解模型：主 -> 备
VISION_MODELS = [
    "hunyuan-vision",
    "qwen3.5-flash",
    "qwen3-vl-plus",
]

# 图像编辑模型：推荐顺序（主 -> 备）
# 这里使用稳定的内部名称，实际请求的模型 ID 由上方配置决定。
MODEL_WAN_27 = "wan2.7-image"
MODEL_QWEN_IMAGE_20 = "qwen-image-2.0"
MODEL_SEEDREAM_50 = "doubao-seedream-5.0"
MODEL_QWEN_EDIT_MAX = "qwen-image-edit-max"

IMAGE_EDIT_MODELS = [
    MODEL_WAN_27,
    MODEL_QWEN_IMAGE_20,
    MODEL_SEEDREAM_50,
    MODEL_QWEN_EDIT_MAX,
]
MODEL_COMPARE_ENABLED_DEFAULT = True
MODEL_COMPARE_LIST = list(IMAGE_EDIT_MODELS)

app = Flask(__name__)
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
    session_path = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_session_data(session_id: str) -> Dict[str, Any]:
    session_path = os.path.join(SESSION_FOLDER, f"{session_id}.json")
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
    rotated = False
    # ESP 屏幕默认按横向展示；遇到竖图先旋转 90 度再做适配。
    if src_h > src_w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        rotated = True
        src_h, src_w = img.shape[:2]

    src_ratio = float(src_w) / float(src_h)
    target_ratio = float(target_w) / float(target_h)

    # 先按比例居中裁剪，再缩放到屏幕分辨率，效果更接近 ImageOps.fit
    if src_ratio > target_ratio:
        crop_w = max(1, int(src_h * target_ratio))
        x0 = max(0, (src_w - crop_w) // 2)
        img = img[:, x0:x0 + crop_w]
    else:
        crop_h = max(1, int(src_w / target_ratio))
        y0 = max(0, (src_h - crop_h) // 2)
        img = img[y0:y0 + crop_h, :]

    interp = cv2.INTER_AREA if (img.shape[1] > target_w or img.shape[0] > target_h) else cv2.INTER_LINEAR
    frame = cv2.resize(img, (target_w, target_h), interpolation=interp)

    jpeg_quality = min(95, max(50, int(quality)))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败，无法发送到屏幕")
    return buf.tobytes(), rotated


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


def evaluate_image_quality(model_name: str, image_url: str) -> Tuple[float, str]:
    img = load_image_from_url_for_quality(image_url)
    if img is None or img.size == 0:
        return -999.0, "图片不可读取"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    col_signal = gray.mean(axis=0)
    row_signal = gray.mean(axis=1)
    col_std = float(np.std(col_signal))
    row_std = float(np.std(row_signal))
    stripe_ratio = col_std / (row_std + 1e-6)
    edges = cv2.Canny(gray, 70, 140)
    edge_density = float(np.mean(edges > 0))

    # 只根据实际输出图评分，不给任何模型预设加分。
    # 评分相同时保留 MODEL_COMPARE_LIST 中靠前的模型。
    score = 0.0

    if lap_var < 12:
        score -= 1.2
    elif lap_var > 950:
        score -= 0.8
    else:
        score += 0.4

    if stripe_ratio > 2.2 and edge_density > 0.07:
        score -= 4.0
        return score, f"疑似条纹伪影 ratio={stripe_ratio:.2f}, edge={edge_density:.3f}"

    if edge_density > 0.24:
        score -= 1.0
    elif edge_density < 0.02:
        score -= 0.6
    else:
        score += 0.3

    return score, f"ok ratio={stripe_ratio:.2f}, edge={edge_density:.3f}, lap={lap_var:.1f}"


# =========================
# Windows ipconfig 提取局域网IP
# =========================
def get_local_ip() -> str:
    """
    强制优先从 Windows 的无线局域网适配器中提取 IPv4。
    找不到再退回以太网，再找不到返回 127.0.0.1
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
        blocks = re.split(r"\n(?=[^\n]*适配器)", text)

        wifi_ip = None
        ethernet_ip = None

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

            if "无线局域网适配器" in title or "wlan" in title or "wi-fi" in title or "wifi" in title:
                wifi_ip = ip
                print(f"[IP识别] 选择无线网卡: {title}")
                print(f"[IP识别] IPv4: {wifi_ip}")
                return wifi_ip

            if "以太网适配器" in title or "ethernet" in title:
                ethernet_ip = ip

        if ethernet_ip:
            print("[IP识别] 未找到无线网卡，改用以太网 IPv4")
            print(f"[IP识别] IPv4: {ethernet_ip}")
            return ethernet_ip

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
    display_url = f"http://{host_ip}:{port}/display"

    mobile_qr_path = os.path.join(STATIC_FOLDER, "qr_mobile.png")
    display_qr_path = os.path.join(STATIC_FOLDER, "qr_display.png")

    generate_qr_image(mobile_url, mobile_qr_path)
    generate_qr_image(display_url, display_qr_path)

    return {
        "mobile_url": mobile_url,
        "display_url": display_url,
        "mobile_qr": "/static/qr_mobile.png",
        "display_qr": "/static/qr_display.png",
    }


# =========================
# 第一次分析提示词
# =========================
def build_analysis_prompt() -> str:
    return """
你是一名专业摄影指导助手。请根据输入照片，完成两件事：

第一，分析当前照片存在的问题，并给出明确、可执行的拍摄建议，这部分内容用于直接展示给用户；
第二，输出一段“ideal_image_prompt”，用于后续图像编辑模型生成一张“调整完成后”的理想拍摄效果图。

请重点完成以下任务：
1. 判断人物在画面中的位置（偏左、偏右、居中，是否过高、过低，留白是否合理）。
2. 判断当前拍摄角度（平拍、俯拍、仰拍、正面、侧面、斜侧等），并说明是否合适。
3. 结合画面中的明暗、阴影、受光面、背景高光、窗户、灯具、太阳方向等线索，判断主要光照源方向是否明确：
   - 如果能够明确判断，就输出主要光照源方向及依据；
   - 如果光源位置不明显、明暗关系不明显、无法可靠判断，就明确写“光源方向不明显”或“无法明确判断”，不要强行推测。
4. 给出摄影师下一步应该如何调整拍摄位置和机位。
5. 给出人物下一步应该如何调整位置和姿态。
6. 判断主体人物的明显性别呈现特征，并输出 subject_gender，仅可填写“男性”或“女性”。
7. 额外输出一段 ideal_image_prompt，用于描述“按照建议调整后，理想拍摄结果应该呈现出的画面”。

请严格只输出 JSON，不要输出解释文字，不要加 markdown 代码块。
JSON 结构必须完全遵循下面的 key：

{
  "scene_summary": "对当前画面的简短概括，1-2句话",
  "subject_gender": "男性或女性",
  "subject_position_analysis": "人物在当前画面中的位置分析",
  "camera_angle_analysis": "当前拍摄角度分析",
  "light_source_inference": "主要光照源方向及依据；若无法判断则明确写光源方向不明显",
  "recommended_shooting_position": "摄影师下一步的机位和站位建议",
  "suggested_adjustment": "人物和相机分别应如何调整",
  "ideal_image_prompt": "用于后续图片编辑的理想真实画面描述"
}

其中，ideal_image_prompt 必须满足以下要求：
1. 它描述的不是当前照片，而是“按建议调整完成后”的理想拍摄结果；
2. 它不是建议，不是分析，不是教学说明，而是一段纯粹的画面内容描述；
3. 它必须重点描述：
   - 人物在画面中的理想位置；
   - 人物的理想拍摄角度；
   - 最终画面的整体观感；
4. 只有在 light_source_inference 可以明确判断光源方向时，才在 ideal_image_prompt 中描述光线方向；如果光源方向不明显，则不要在 ideal_image_prompt 中强调具体光源位置；
5. 不要详细描述环境内容，因为后续图像编辑模型会直接参考原图环境；
6. 人物性别必须与 subject_gender 一致；
7. ideal_image_prompt 描述的是“真实照片效果”，不是卡通图，不是插画，不是示意图；
8. 不要加入箭头、图标、标注、气泡框、教学说明、对话框、海报文字；
9. 不要写成“应该如何拍”，而要写成“画面最终是什么样子”；
10. 输出必须是中文，适合直接给图像编辑模型使用。

要求：
- 输出必须是中文；
- 结果必须适合网页直接展示；
- 对人物性别和人物身份特征要尽量依据照片中的真实外观，不要随意改变；
- 如果光照信息不足，不要虚构原图中不存在的光源。
""".strip()

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
def call_vision_once(model_name: str, image_path: str) -> Dict[str, str]:
    if model_name.startswith("hunyuan"):
        return call_vision_hunyuan_once(model_name, image_path)
    image_data_url = image_file_to_data_url(image_path)
    prompt = build_analysis_prompt()

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


def call_vision_hunyuan_once(model_name: str, image_path: str) -> Dict[str, str]:
    # 检查 Key 格式
    if HUNYUAN_API_KEY and HUNYUAN_API_KEY.startswith("AKID"):
        print("!!! 警告: HUNYUAN_API_KEY 看起来像是腾讯云 SecretId (以 AKID 开头)。")
        print("!!! 混元 OpenAI 兼容接口需要单独的 API Key，而非 SecretId/SecretKey。")
        print("!!! 请前往 https://console.cloud.tencent.com/hunyuan/start 获取 API Key。")

    image_data_url = image_file_to_data_url(image_path)
    prompt = build_analysis_prompt()
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


def check_vision_model_ready(model_name: str) -> Tuple[bool, str]:
    if model_name.startswith("hunyuan"):
        return bool(HUNYUAN_API_KEY), "未配置 HUNYUAN_API_KEY"
    return bool(DASHSCOPE_API_KEY), "未配置 DASHSCOPE_API_KEY"


def call_vision_auto(image_path: str) -> Tuple[Dict[str, str], str]:
    last_error: Optional[Exception] = None
    ready_models: List[str] = []
    for model_name in VISION_MODELS:
        ready, _ = check_vision_model_ready(model_name)
        if not ready:
            continue
        ready_models.append(model_name)
        try:
            result = call_vision_once(model_name, image_path)
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
    visual_goal = str(raw_intent.get("visual_goal", "")).strip() or "更自然耐看"
    emotion_style = str(raw_intent.get("emotion_style", "")).strip() or "平静"
    usage_type = str(raw_intent.get("usage_type", "")).strip() or "个人写真"
    lens_preference = str(raw_intent.get("lens_preference", "")).strip() or "更自然视角"
    return {
        "visual_goal": visual_goal,
        "emotion_style": emotion_style,
        "usage_type": usage_type,
        "lens_preference": lens_preference,
    }


def build_expression_hint(emotion_style: str) -> str:
    mapping = {
        "快乐": "建议轻微上扬嘴角，眼神聚焦镜头并带轻微神采",
        "平静": "建议保持自然中性表情，眼神聚焦镜头，呼吸放松",
        "惆怅": "建议嘴角轻微收敛，视线略微放空",
        "神秘": "建议表情克制，目光轻微偏离镜头",
        "温暖": "建议微笑不露齿，眼神柔和且有微光",
        "冷感": "建议表情平稳，目光清冷但保持聚焦",
    }
    return mapping.get(emotion_style, "建议保持自然表情，眼神聚焦镜头并轻微提神")


def build_composition_methods(user_intent: Dict[str, str]) -> str:
    visual_goal = user_intent.get("visual_goal", "")
    lens_pref = user_intent.get("lens_preference", "")
    usage_type = user_intent.get("usage_type", "")
    methods: List[str] = ["三分法定位主体", "留白控制画面呼吸感", "水平垂直线校正"]

    if visual_goal == "背景更壮观":
        methods = ["引导线强化纵深", "前景-主体-背景三层结构", "对角线构图扩展空间感", "三分法与留白协同"]
    elif visual_goal == "人物更突出":
        methods = ["框架构图聚焦主体", "色彩对比突出人物", "背景简化与留白收敛", "三分法稳定主体重心"]
    elif visual_goal == "画面更干净":
        methods = ["留白构图减噪", "对称或准对称提升秩序感", "主体边缘干扰清理"]
    elif visual_goal == "更像电影镜头":
        methods = ["前景虚化营造氛围", "引导线建立叙事动线", "对角线与留白塑造镜头感"]
    elif visual_goal == "显腿更长":
        methods = ["低机位对角线延展腿部", "三角形站姿稳定重心", "前景弱化避免腿部遮挡"]

    if lens_pref == "更广角冲击":
        methods.append("近大远小强化透视")
    elif lens_pref == "更压缩背景":
        methods.append("中长焦压缩减少背景干扰")
    elif lens_pref == "更突出人物":
        methods.append("主体占比优先并控制边缘杂物")

    if usage_type == "形象照":
        methods.append("肩线与头部微三角构图增强松弛感")

    return "；".join(methods)


def build_strategy_plan(ai_result: Dict[str, str], user_intent: Dict[str, str]) -> Dict[str, str]:
    strategy = {
        "composition_goal": "保持主体清晰与画面平衡",
        "camera_height": "平视",
        "camera_angle": "正面或轻侧面",
        "shot_size_target": "半身到全身之间",
        "lens_feel": "自然视角",
        "subject_position_target": "人物靠近三分线或轻微居中",
        "subject_ratio_target": "人物占画面约35%到50%",
        "negative_space_target": "保留适度留白，避免过空",
        "light_strategy": "仅利用原图已有光线，无法判断方向时不新增光源",
        "depth_strategy": "保持自然纵深",
        "emotion_style": user_intent["emotion_style"],
        "pose_strategy": "姿态自然放松，身体轻微转向",
        "expression_hint": build_expression_hint(user_intent["emotion_style"]),
        "eye_focus_strategy": "双眼注视镜头附近，保留清晰虹膜边界与轻微眼部高光，避免呆滞眼神",
        "composition_methods": build_composition_methods(user_intent),
        "color_strategy": "保持真实肤色稍微提亮增白，利用主体与背景色温/明度对比突出人物",
        "geometry_strategy": "校正地平线与垂直线，避免画面倾斜和空间变形",
    }

    visual_goal = user_intent["visual_goal"]
    if visual_goal == "显腿更长":
        strategy.update({
            "composition_goal": "强化腿部延展感并保持自然比例",
            "camera_height": "低机位",
            "camera_angle": "轻微仰拍",
            "shot_size_target": "优先全身",
            "lens_feel": "自然偏广角",
            "subject_position_target": "人物位于画面中下区域",
            "subject_ratio_target": "人物占画面约45%到60%",
            "pose_strategy": "前后脚错步，重心自然前移",
        })
    elif visual_goal == "人物更突出":
        strategy.update({
            "composition_goal": "压缩干扰元素，突出人物主体",
            "shot_size_target": "半身或近景",
            "lens_feel": "中焦感",
            "subject_ratio_target": "人物占画面约50%到70%",
            "negative_space_target": "减少无效留白",
            "depth_strategy": "弱化背景存在感",
        })
    elif visual_goal == "背景更壮观":
        strategy.update({
            "composition_goal": "强化人物与环境的尺度关系",
            "shot_size_target": "环境人像",
            "lens_feel": "广角感",
            "subject_position_target": "人物位于下三分之一或侧三分线",
            "subject_ratio_target": "人物占画面约20%到35%",
            "negative_space_target": "适度增加环境留白",
            "depth_strategy": "增强远近层次",
        })
    elif visual_goal == "画面更干净":
        strategy.update({
            "composition_goal": "清除干扰，保持简洁构图",
            "subject_position_target": "人物稳定居中或三分线",
            "negative_space_target": "留白集中且可解释",
            "depth_strategy": "减少杂乱层次",
        })
    elif visual_goal == "更像电影镜头":
        strategy.update({
            "composition_goal": "强化叙事感与留白结构",
            "camera_angle": "正面或斜侧",
            "shot_size_target": "中景偏环境",
            "negative_space_target": "保留更多叙事留白",
            "depth_strategy": "增强前中后层次",
        })

    lens_pref = user_intent["lens_preference"]
    if lens_pref == "更广角冲击":
        strategy["lens_feel"] = "广角感"
        strategy["depth_strategy"] = "增强空间纵深"
        strategy["camera_angle"] = "轻微仰拍或斜侧"
        strategy["subject_ratio_target"] = "人物占画面约28%到45%"
        strategy["negative_space_target"] = "保留可读的环境留白并增强透视延展"
        strategy["composition_methods"] = strategy.get("composition_methods", "") + "；近大远小透视强化；前景锚点增强广角冲击"
    elif lens_pref == "更突出人物":
        strategy["lens_feel"] = "中焦感"
        strategy["subject_ratio_target"] = "人物占画面约55%到72%"
    elif lens_pref == "更压缩背景":
        strategy["lens_feel"] = "长焦压缩感"
        strategy["depth_strategy"] = "压缩背景层次"

    emotion_style = user_intent["emotion_style"]
    if emotion_style == "快乐":
        strategy["pose_strategy"] = "姿态打开，视线更积极"
    elif emotion_style == "惆怅":
        strategy["pose_strategy"] = "视线轻微偏离镜头，动作克制"
    elif emotion_style == "神秘":
        strategy["pose_strategy"] = "保留侧向轮廓，减少正面直视"
    elif emotion_style == "温暖":
        strategy["pose_strategy"] = "表情柔和，动作自然靠近"
    elif emotion_style == "冷感":
        strategy["pose_strategy"] = "姿态稳定，表情克制"

    if "不明显" in ai_result.get("light_source_inference", ""):
        strategy["light_strategy"] = "光源方向不明显，保持原图自然明暗关系，不制造新光源"

    return strategy


def build_targeted_ideal_prompt(
    ai_result: Dict[str, str],
    strategy_plan: Dict[str, str],
    user_intent: Dict[str, str]
) -> str:
    light_text = ai_result.get("light_source_inference", "")
    light_clause = "整体光线保持原图自然关系。"
    if "不明显" not in light_text and "无法" not in light_text:
        light_clause = "利用原图已有的受光方向优化面部与主体明暗层次，不新增任何新光源。"

    return (
        f"这是一张真实照片风格的人像优化结果。人物以{strategy_plan['shot_size_target']}呈现，"
        f"机位为{strategy_plan['camera_height']}，角度采用{strategy_plan['camera_angle']}，"
        f"画面构图目标是{strategy_plan['composition_goal']}，人物位置目标为{strategy_plan['subject_position_target']}。"
        f"镜头观感为{strategy_plan['lens_feel']}，空间层次策略为{strategy_plan['depth_strategy']}，"
        f"构图执行方法包括{strategy_plan.get('composition_methods', '三分法与留白控制')}。"
        f"情绪表达为{user_intent['emotion_style']}，用途偏向{user_intent['usage_type']}。"
        f"{light_clause} 眼神策略为{strategy_plan.get('eye_focus_strategy', '眼神自然聚焦')}。"
        f"人物状态自然上镜，保持同一人物身份特征，环境维持原场景。"
        "Identity lock: keep the same person and same outfit/hair/accessories; "
        "do not swap face/person; do not add or remove major scene elements."
    )


def build_guidance_text(
    ai_result: Dict[str, str],
    strategy_plan: Dict[str, str],
    user_intent: Dict[str, str]
) -> Dict[str, str]:
    return {
        "scene_summary": ai_result.get("scene_summary", "已完成基础诊断。"),
        "subject_gender": ai_result.get("subject_gender", "未返回该项结果。"),
        "subject_position_analysis": ai_result.get("subject_position_analysis", "未返回该项结果。"),
        "camera_angle_analysis": ai_result.get("camera_angle_analysis", "未返回该项结果。"),
        "shot_size_analysis": ai_result.get("shot_size_analysis", "未返回该项结果。"),
        "composition_analysis": ai_result.get("composition_analysis", "未返回该项结果。"),
        "light_source_inference": ai_result.get("light_source_inference", "未返回该项结果。"),
        "recommended_shooting_position": (
            f"目标为“{user_intent['visual_goal']}”，建议{strategy_plan['camera_height']} + "
            f"{strategy_plan['camera_angle']}，并保持{strategy_plan['subject_position_target']}。"
        ),
        "suggested_adjustment": (
            f"摄影师：按{strategy_plan['lens_feel']}与{strategy_plan['shot_size_target']}执行；"
            f"人物：{strategy_plan['pose_strategy']}；情绪方向：{strategy_plan['emotion_style']}；"
            f"表情建议：{strategy_plan.get('expression_hint', '保持自然表情，轻微调整即可')}；"
            f"眼神策略：{strategy_plan.get('eye_focus_strategy', '眼神自然聚焦')}；"
            f"构图方法：{strategy_plan.get('composition_methods', '三分法+留白')}。"
        ),
        "expression_hint": strategy_plan.get("expression_hint", "保持自然表情，轻微调整即可"),
        "ideal_image_prompt": ai_result.get("ideal_image_prompt", "未返回该项结果。"),
    }


def build_edit_prompt(
    ai_result: Dict[str, str],
    strategy_plan: Dict[str, str],
    user_intent: Dict[str, str],
    targeted_ideal_prompt: str
) -> str:
    _ = strategy_plan, user_intent, targeted_ideal_prompt
    return f"""
请基于输入原图进行写实照片风格编辑，生成“调整后的理想拍摄结果图”。

总原则（必须同时满足）：
1. 人物身份锁定：必须是同一人，不换脸、不换人，不改变年龄感、性别、体型、发型、服装、配饰。
2. 场景内容锁定：原图中的背景、道具、空间关系与关键元素必须保留，不新增、不删除、不替换任何主要元素。
3. 光线约束：仅可利用原图已有光线关系，不凭空添加新光源。
4. 风格约束：保持真实摄影质感，不要卡通、插画、动漫、海报字、气泡框、教学标注。

允许AI发挥的范围（请大胆执行）：
1. 允许进行明显的机位变化（平拍、俯拍、仰拍、斜侧拍）、取景重构与构图重排。
2. 允许调整人物站位、身体朝向、头肩角度、手臂与躯干姿态，使画面更有表现力。
3. 允许在不改变场景元素的前提下做较大幅度裁切、透视重构、主体比例调整。
4. 允许做自然且克制的人像优化（提神、肤质微调、层次增强），但不得出现整容感或失真。

请重点追求：
- 更有创意的拍摄角度与人物姿态；
- 但画面“内容元素”保持和原图一致；
- 最终看起来像真实相机拍摄结果。

人物性别：
{ai_result.get("subject_gender", "")}

理想拍摄结果描述：
{ai_result.get("ideal_image_prompt", "")}

光线判断参考：
{ai_result.get("light_source_inference", "")}

反向限制：
- 不要换脸、不要换人
- 不要改变穿着、发型、配饰
- 不要新增或删除原图主要元素
- 不要新增原图不存在的光源
- 不要卡通/插画/动漫化
- 不要任何文字标注或对话框
- 不要过度磨皮、不要失真
""".strip()

QWEN_IMAGE_NEGATIVE_PROMPT = (
    "微笑,大笑,露齿,张嘴,改变表情,表情变形,卡通风格,插画风格,动漫风格,二次元,"
    "气泡对话框,对白框,教学标注,箭头,标签,海报标题,大段文字,遮挡主体的文字,陌生人,"
    "性别错误,明显换脸,化妆,眼影,口红,美妆,脂粉气,浓妆,烟熏妆,条纹噪点,网纹,摩尔纹,"
    "扫描线,网格噪声,纹理污染,呆滞眼神,空洞眼神,失焦眼神,斗鸡眼"
)


def edit_with_dashscope_image_model(
    image_path: str,
    prompt_text: str,
    model_id: str,
    use_negative_prompt: bool,
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
        "size": "1024*1536",
    }
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


def edit_with_wan27_image(image_path: str, prompt_text: str) -> str:
    return edit_with_dashscope_image_model(
        image_path,
        prompt_text,
        WAN_IMAGE_MODEL,
        use_negative_prompt=False,
    )


def edit_with_qwen_image_20(image_path: str, prompt_text: str) -> str:
    return edit_with_dashscope_image_model(
        image_path,
        prompt_text,
        QWEN_IMAGE_MODEL,
        use_negative_prompt=True,
    )


def edit_with_qwen_image_edit(image_path: str, prompt_text: str) -> str:
    return edit_with_dashscope_image_model(
        image_path,
        prompt_text,
        QWEN_IMAGE_EDIT_MAX_MODEL,
        use_negative_prompt=True,
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


def edit_with_doubao_seedream(image_path: str, prompt_text: str) -> str:
    if not DOUBAO_API_KEY:
        raise RuntimeError("未配置 DOUBAO_API_KEY")
    image_data_url = image_file_to_data_url(image_path)
    headers = {
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
        "Content-Type": "application/json",
    }
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
                    "size": "2K",
                    "sequential_image_generation": "disabled",
                    "response_format": "url",
                    "watermark": False,
                })
            else:
                # 兼容用户已配置的旧 /images/edits 接口。
                payload.update({
                    "image": image_data_url,
                    "size": "1024x1536",
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


def run_model_compare(image_path: str, prompt_text: str) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    chosen: Optional[Dict[str, Any]] = None
    errors: List[str] = []

    active_models = get_ready_models(MODEL_COMPARE_LIST)
    for model_name in active_models:
        runner = get_image_model_runner(model_name)
        if runner is None:
            continue
        try:
            raw_url = runner(image_path, prompt_text)
            if not str(raw_url).strip() or not (
                str(raw_url).startswith("http://")
                or str(raw_url).startswith("https://")
                or str(raw_url).startswith("/uploads/")
            ):
                raise RuntimeError("模型返回空图片地址。")
            raw_local_url, beauty_url, beauty_applied = raw_url, raw_url, False
            quality_score, quality_reason = evaluate_image_quality(model_name, beauty_url)
            item = {
                "model": model_name,
                "raw_diagram_url": raw_local_url,
                "diagram_url": beauty_url,
                "beauty_applied": beauty_applied,
                "error": "",
                "quality_score": quality_score,
                "quality_reason": quality_reason,
            }
            candidates.append(item)
        except Exception as e:
            err = str(e)
            if not err and isinstance(e, requests.HTTPError):
                try:
                    err = safe_resp_text(e.response)
                except Exception:
                    err = ""
            pretty_err = err
            if "InternalError" in err or "submit algo service error" in err:
                pretty_err = "服务繁忙（InternalError），已自动回退主模型"
            candidates.append({
                "model": model_name,
                "raw_diagram_url": "",
                "diagram_url": "",
                "beauty_applied": False,
                "error": pretty_err,
                "quality_score": -999.0,
                "quality_reason": "",
            })
            errors.append(f"{model_name} 失败：{pretty_err}")

    successful_items = [x for x in candidates if not x.get("error")]
    valid_items = [
        x for x in successful_items
        if "条纹伪影" not in str(x.get("quality_reason", ""))
        and float(x.get("quality_score", -999.0)) > -3.0
    ]
    if valid_items:
        # max 在同分时保留列表中第一项，即优先遵循推荐顺序。
        chosen = max(valid_items, key=lambda z: float(z.get("quality_score", -999.0)))
    elif successful_items:
        # 质量检测无法读取远程图时仍保留成功结果，避免误判为生成失败。
        chosen = max(successful_items, key=lambda z: float(z.get("quality_score", -999.0)))

    if chosen:
        return {
            "diagram_url": chosen["diagram_url"],
            "raw_diagram_url": chosen["raw_diagram_url"],
            "image_model": chosen["model"],
            "diagram_error": "",
            "beauty_applied": chosen["beauty_applied"],
            "model_candidates": candidates,
        }

    return {
        "diagram_url": "",
        "raw_diagram_url": "",
        "image_model": "",
        "diagram_error": "；".join(errors),
        "beauty_applied": False,
        "model_candidates": candidates,
    }


def edit_image_auto(image_path: str, prompt_text: str) -> Tuple[str, str, str]:
    errors: List[str] = []
    for model_name in get_ready_models(IMAGE_EDIT_MODELS):
        runner = get_image_model_runner(model_name)
        if runner is None:
            continue
        try:
            return runner(image_path, prompt_text), model_name, ""
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
    return render_template("mobile.html")


@app.route("/mobile")
def mobile_page():
    return render_template("mobile.html")


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
        esp_ip = str(payload.get("esp_ip", "")).strip()
        if not esp_ip:
            return jsonify({"ok": False, "msg": "请填写 ESP 屏幕 IP（例如 192.168.1.50）"}), 400
        if not is_valid_ipv4(esp_ip):
            return jsonify({"ok": False, "msg": "ESP IP 格式不正确，请填写 IPv4 地址"}), 400

        update_api_keys_file({"ESP_SCREEN_IP": esp_ip})
        cfg = get_runtime_esp_config()
        return jsonify({
            "ok": True,
            "msg": f"已保存 ESP 屏幕 IP：{esp_ip}",
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

        ai_result, used_vision_model = call_vision_auto(save_path)
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

        session_data = load_session_data(session_id)
        image_path = session_data.get("image_path", "")
        if not image_path or not os.path.exists(image_path):
            return jsonify({"ok": False, "msg": "会话原图不存在，请重新拍照。"}), 400

        diagnosis_report = session_data.get("diagnosis_report", {})
        if not isinstance(diagnosis_report, dict):
            diagnosis_report = {}

        user_intent = normalize_user_intent(payload.get("user_intent", {}))
        debug_compare = bool(payload.get("debug_compare", MODEL_COMPARE_ENABLED_DEFAULT))
        strategy_plan = build_strategy_plan(diagnosis_report, user_intent)
        targeted_ideal_prompt = build_targeted_ideal_prompt(diagnosis_report, strategy_plan, user_intent)
        edit_prompt = build_edit_prompt(diagnosis_report, strategy_plan, user_intent, targeted_ideal_prompt)
        final_guidance = build_guidance_text(diagnosis_report, strategy_plan, user_intent)
        final_guidance["ideal_image_prompt"] = targeted_ideal_prompt

        print("[开始图像编辑生成理想拍摄结果图]")
        model_candidates: List[Dict[str, Any]] = []
        if debug_compare:
            compare_data = run_model_compare(image_path, edit_prompt)
            diagram_url = compare_data.get("diagram_url", "")
            used_image_model = compare_data.get("image_model", "")
            diagram_error = compare_data.get("diagram_error", "")
            raw_diagram_url = compare_data.get("raw_diagram_url", diagram_url)
            beauty_applied = bool(compare_data.get("beauty_applied", False))
            model_candidates = compare_data.get("model_candidates", [])
        else:
            diagram_url, used_image_model, diagram_error = edit_image_auto(image_path, edit_prompt)
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
            latest_data["esp_push_ok"] = ok
            latest_data["esp_push_msg"] = msg
            latest_data["esp_pushed_image"] = image_ref if ok else ""
            latest_data["esp_push_count"] = success_count
            latest_data["esp_push_total"] = len(queue)
            save_latest_result(latest_data)

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
    qrcode_url = f"http://{host_ip}:5000/qrcode"

    print("====================================")
    print("服务已准备启动")
    print(f"手机拍照页：{qr_info['mobile_url']}")
    print(f"电脑显示页：{qr_info['display_url']}")
    print(f"二维码页：{qrcode_url}")
    print(f"视觉理解模型顺序：{VISION_MODELS}")
    print(f"图像编辑模型顺序：{get_ready_models(IMAGE_EDIT_MODELS)}")
    print("二维码图片已生成：")
    print(f"  {os.path.join(STATIC_FOLDER, 'qr_mobile.png')}")
    print(f"  {os.path.join(STATIC_FOLDER, 'qr_display.png')}")
    print("====================================")

    # 启动后自动打开二维码页
    open_browser_later(qrcode_url, delay=1.2)

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
