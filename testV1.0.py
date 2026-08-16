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

# Qwen 图像编辑原生接口
MULTIMODAL_GEN_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# 智谱图像配置
ZHIPU_API_KEY = load_env("ZHIPU_API_KEY")
ZHIPU_IMAGE_URL = load_env("ZHIPU_IMAGE_URL", "https://open.bigmodel.cn/api/paas/v4/images/generations")
ZHIPU_IMAGE_MODEL = load_env("ZHIPU_IMAGE_MODEL", "glm-image")

# 豆包图像配置
DOUBAO_API_KEY = load_env("DOUBAO_API_KEY")
DOUBAO_IMAGE_URL = load_env("DOUBAO_IMAGE_URL", "https://ark.cn-beijing.volces.com/api/v3/images/edits")
DOUBAO_IMAGE_MODEL = load_env("DOUBAO_IMAGE_MODEL", "Doubao-Seedream-5.0-lite")
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

# 图像编辑模型：主 -> 备
IMAGE_EDIT_MODELS = [
    "qwen-image-edit-max",
    "Doubao-Seedream-5.0-lite",
    "jimeng-ai-image-4.0",
]
MODEL_COMPARE_ENABLED_DEFAULT = True
MODEL_COMPARE_LIST = [
    "qwen-image-edit-max",
    "Doubao-Seedream-5.0-lite",
    "jimeng-ai-image-4.0",
]

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
    user_candidates = [
        str(DOUBAO_IMAGE_MODEL or "").strip(),
        str(DOUBAO_ENDPOINT_ID or "").strip(),
    ]
    expanded_user_candidates: List[str] = []
    for item in user_candidates:
        if not item:
            continue
        expanded_user_candidates.append(item)
        low = item.lower()
        if ("seedream" in low or "doubao" in low) and low != item:
            expanded_user_candidates.append(low)
    fetched_ids = list_doubao_model_ids()
    ranked_ids: List[str] = []
    for mid in fetched_ids:
        low = mid.lower()
        if low.startswith("ep-"):
            ranked_ids.append(mid)
    for mid in fetched_ids:
        low = mid.lower()
        if ("seedream" in low or "image" in low) and mid not in ranked_ids:
            ranked_ids.append(mid)
    seen: set = set()
    ordered: List[str] = []
    for item in expanded_user_candidates + ranked_ids:
        if not item or item in seen:
            continue
        ordered.append(item)
        seen.add(item)
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


def build_screen_jpeg(image_bytes: bytes, target_w: int, target_h: int, quality: int) -> bytes:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise RuntimeError("图片解码失败，无法发送到屏幕")

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
    return buf.tobytes()


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
        jpg_bytes = build_screen_jpeg(raw_bytes, width, height, q)
        resp = requests.post(
            url,
            files={"file": ("frame.jpg", jpg_bytes, "image/jpeg")},
            timeout=max(1.0, float(cfg["timeout_sec"])),
        )
        if resp.status_code != 200:
            body = (resp.text or "").strip()
            if resp.status_code == 400 and "STORE_FAILED" in body.upper():
                return False, (
                    f"ESP 存储失败（HTTP 400: {body[:120]}）。"
                    "请确认固件按 ESP32-S3 OPI PSRAM 模式编译（PSRAM=opi），当前图片 "
                    f"{len(jpg_bytes)} bytes, q={q}"
                )
            return False, f"ESP 推送失败 HTTP {resp.status_code}: {body[:120]} (q={q}, {len(jpg_bytes)} bytes)"
        return True, f"ESP 推送成功：{url} ({len(jpg_bytes)} bytes, q={q})"
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

    score = 0.0
    if model_name == "qwen-image-edit-max":
        score += 1.2
    elif model_name == "doubao-Seedream-5-0-260128":
        score += 0.7
    elif model_name == "zhipu-glm-image":
        score += 0.6
    elif model_name == "jimeng-ai-image-4.0":
        score += 0.5

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
你是一名专业摄影指导助手。请先做客观诊断，再给基础建议，不要先做风格化发挥。

请严格只输出 JSON，不要输出解释文字，不要加 markdown 代码块。
JSON 结构必须完全遵循下面的 key：

{
  "scene_summary": "对当前画面的简短概括，1-2句话",
  "subject_gender": "男性或女性",
  "subject_position_analysis": "人物在当前画面中的位置分析",
  "camera_angle_analysis": "当前拍摄角度分析",
  "shot_size_analysis": "当前景别分析，判断远景/全身/半身/特写是否匹配",
  "composition_analysis": "构图问题诊断，重点判断头顶留白、人物偏移、前景遮挡",
  "light_source_inference": "主要光照源方向及依据；若无法判断则明确写光源方向不明显",
  "recommended_shooting_position": "摄影师下一步的机位和站位建议",
  "suggested_adjustment": "人物和相机分别应如何调整，保持客观、可执行",
  "ideal_image_prompt": "用于后续图片编辑的基础理想画面描述，不要加入风格化词汇"
}

要求：
- 光源只有明确可判断时才写方向，不明确时必须写“光源方向不明显”；
- 基础诊断阶段不做“电影感、高级感”等风格化结论；
- ideal_image_prompt 描述真实照片效果，保持原场景，不要新增原图不存在的光源；
- 不要加入气泡框、箭头、标注、海报文字；
- 输出必须是中文。
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


def build_style_technique_note(user_intent: Dict[str, str]) -> str:
    visual_goal = user_intent.get("visual_goal", "")
    emotion_style = user_intent.get("emotion_style", "")
    usage_type = user_intent.get("usage_type", "")
    lens_pref = user_intent.get("lens_preference", "")

    visual_notes = {
        "更自然耐看": "主技法：平视或轻侧面，控制头顶留白和肩线，保持真实生活感。",
        "人物更突出": "主技法：提高主体占比并弱化背景干扰，眼睛和面部层次优先。",
        "背景虚化分离": "主技法：轻微浅景深分离，背景柔化但不抹掉原场景。",
        "背景更壮观": "主技法：保留环境范围，用人物与背景尺度对比增强空间感。",
        "引导线纵深": "主技法：用原图已有线条引导视线，不强行新增道路或结构。",
        "框架构图": "主技法：利用原图已有门窗、墙边或前景边缘形成自然框架。",
        "前景层次": "主技法：保留少量不遮挡脸部的前景虚化，形成空间层次。",
        "画面更干净": "主技法：减少边缘杂物和无效留白，不凭空删除关键场景元素。",
        "对称稳定": "主技法：校正水平垂直关系，让人物重心稳定。",
        "低角度张力": "主技法：轻微低机位增强张力，避免脸部和四肢变形。",
        "显腿更长": "主技法：低机位和对角线轻微延展腿部，比例必须自然。",
        "逆光轮廓": "主技法：仅在原图已有背光或边缘亮部时轻微增强轮廓。",
        "更像电影镜头": "主技法：叙事留白和前中后景层次，保持真实照片质感。",
    }

    emotion_notes = {
        "平静": "情绪：中性放松。",
        "温暖": "情绪：柔和亲近。",
        "快乐": "情绪：轻微积极，避免夸张大笑。",
        "自信": "情绪：肩线打开，眼神稳定。",
        "松弛": "情绪：自然不摆拍。",
        "专注": "情绪：眼神聚焦，表情克制。",
        "清爽": "情绪：明暗通透，避免厚重滤镜。",
        "惆怅": "情绪：轻微偏离镜头，保留呼吸感。",
        "神秘": "情绪：可有局部阴影，但眼部必须可读。",
        "冷感": "情绪：色彩克制，轮廓清楚。",
        "高反差戏剧": "情绪：只利用原图已有明暗强化张力。",
    }

    usage_notes = {
        "职业头像": "用途：胸像或半身，背景干净，肩线端正。",
        "社交头像": "用途：人物识别度高，表情自然亲近。",
        "街拍人像": "用途：保留环境线索和现场感。",
        "夜景人像": "用途：保留原有环境灯光，控制脸部暗部。",
        "校园记录": "用途：真实纪实，清爽不过度修饰。",
        "毕业纪念": "用途：端正、有纪念感，不海报化。",
        "形象照": "用途：精神状态、肩颈线和面部清晰度优先。",
    }

    lens_notes = {
        "更自然视角": "镜头：标准透视。",
        "35mm环境人像": "镜头：35mm 环境人像感，保留环境。",
        "50mm标准纪实": "镜头：50mm 标准纪实感，比例稳定。",
        "85mm半身人像": "镜头：85mm 半身人像感，背景轻微柔化。",
        "长焦浅景深": "镜头：长焦浅景深，眼睛和人物边缘必须清晰。",
        "更广角冲击": "镜头：广角纵深，避免夸张变形。",
        "低机位广角": "镜头：轻微低机位广角，控制比例自然。",
        "更突出人物": "镜头：主体占比优先。",
        "更压缩背景": "镜头：中长焦压缩背景层次。",
        "前景虚化层次": "镜头：前景轻微虚化，不遮挡脸部。",
    }

    parts = [visual_notes.get(visual_goal, ""), lens_notes.get(lens_pref, "")]
    soft_parts = [emotion_notes.get(emotion_style, ""), usage_notes.get(usage_type, "")]
    note = " ".join([p for p in parts + soft_parts if p]).strip()
    return (
        f"{note} 执行强度：克制自然，只做一档优化；只能利用原图已经存在的线条、前景、背景、光源和道具；"
        "以原图真实内容为准，不要机械叠加全部技法，不要新增原图不存在的元素。"
    ).strip()


def build_expression_hint(emotion_style: str) -> str:
    mapping = {
        "快乐": "建议轻微上扬嘴角，眼神聚焦镜头并带轻微神采",
        "平静": "建议保持自然中性表情，眼神聚焦镜头，呼吸放松",
        "惆怅": "建议嘴角轻微收敛，视线略微放空",
        "神秘": "建议表情克制，目光轻微偏离镜头",
        "温暖": "建议微笑不露齿，眼神柔和且有微光",
        "冷感": "建议表情平稳，目光清冷但保持聚焦",
        "自信": "建议肩线打开，眼神稳定看向镜头，表情从容",
        "松弛": "建议放松肩颈和嘴角，眼神自然不过度用力",
        "专注": "建议眼神聚焦明确，表情克制，减少多余动作",
        "清爽": "建议表情自然轻盈，眼神清晰，避免厚重情绪",
        "高反差戏剧": "建议表情克制，眼神有力量，利用原有明暗形成张力",
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
    elif visual_goal == "背景虚化分离":
        methods = ["主体边缘清晰", "背景柔化减噪", "眼部焦点优先"]
    elif visual_goal == "引导线纵深":
        methods = ["利用原有线条指向主体", "保持水平垂直自然", "远近层次递进"]
    elif visual_goal == "框架构图":
        methods = ["利用原有门窗或边缘框架", "主体落在框架内", "控制边缘遮挡"]
    elif visual_goal == "前景层次":
        methods = ["前景轻微虚化", "主体脸部无遮挡", "前景-主体-背景三层结构"]
    elif visual_goal == "画面更干净":
        methods = ["留白构图减噪", "对称或准对称提升秩序感", "主体边缘干扰清理"]
    elif visual_goal == "对称稳定":
        methods = ["中轴或准中轴构图", "水平垂直线校正", "左右视觉重量平衡"]
    elif visual_goal == "低角度张力":
        methods = ["轻微低机位", "对角线增强张力", "控制脸部与四肢变形"]
    elif visual_goal == "更像电影镜头":
        methods = ["前景虚化营造氛围", "引导线建立叙事动线", "对角线与留白塑造镜头感"]
    elif visual_goal == "显腿更长":
        methods = ["低机位对角线延展腿部", "三角形站姿稳定重心", "前景弱化避免腿部遮挡"]
    elif visual_goal == "逆光轮廓":
        methods = ["保留原有背光关系", "轻微增强人物边缘亮部", "避免凭空新增强光"]

    if lens_pref == "更广角冲击":
        methods.append("近大远小强化透视")
    elif lens_pref == "低机位广角":
        methods.append("轻微低机位广角但控制比例自然")
    elif lens_pref in {"35mm环境人像", "50mm标准纪实"}:
        methods.append("保留环境线索与自然透视")
    elif lens_pref in {"85mm半身人像", "长焦浅景深"}:
        methods.append("眼部清晰并柔化背景干扰")
    elif lens_pref == "更压缩背景":
        methods.append("中长焦压缩减少背景干扰")
    elif lens_pref == "更突出人物":
        methods.append("主体占比优先并控制边缘杂物")
    elif lens_pref == "前景虚化层次":
        methods.append("前景虚化只在边缘形成层次")

    if usage_type == "形象照":
        methods.append("肩线与头部微三角构图增强松弛感")
    elif usage_type in {"职业头像", "社交头像"}:
        methods.append("眼神与面部清晰度优先")
    elif usage_type in {"夜景人像", "街拍人像"}:
        methods.append("保留现场环境氛围")

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
        "technique_note": build_style_technique_note(user_intent),
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
    elif visual_goal == "背景虚化分离":
        strategy.update({
            "composition_goal": "主体清晰并与背景柔和分离",
            "shot_size_target": "半身或中近景",
            "lens_feel": "中长焦浅景深感",
            "subject_ratio_target": "人物占画面约45%到62%",
            "depth_strategy": "背景柔化但保留原场景信息",
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
    elif visual_goal == "引导线纵深":
        strategy.update({
            "composition_goal": "利用原图线条建立视线纵深",
            "shot_size_target": "中景偏环境",
            "subject_position_target": "人物靠近线条汇聚方向或三分线",
            "depth_strategy": "增强远近递进关系",
        })
    elif visual_goal == "框架构图":
        strategy.update({
            "composition_goal": "用原有边缘元素形成自然框架",
            "subject_position_target": "人物位于框架内或三分线附近",
            "negative_space_target": "保留框架边缘但不遮挡脸部",
        })
    elif visual_goal == "前景层次":
        strategy.update({
            "composition_goal": "建立前景、人物、背景三层空间",
            "shot_size_target": "中景或半身",
            "depth_strategy": "前景轻微虚化，主体清晰",
        })
    elif visual_goal == "画面更干净":
        strategy.update({
            "composition_goal": "清除干扰，保持简洁构图",
            "subject_position_target": "人物稳定居中或三分线",
            "negative_space_target": "留白集中且可解释",
            "depth_strategy": "减少杂乱层次",
        })
    elif visual_goal == "对称稳定":
        strategy.update({
            "composition_goal": "建立稳定中轴或准对称构图",
            "subject_position_target": "人物居中或准居中",
            "negative_space_target": "左右留白均衡",
            "geometry_strategy": "优先校正水平线、垂直线和中轴关系",
        })
    elif visual_goal == "低角度张力":
        strategy.update({
            "composition_goal": "轻微低机位增强张力但保持比例自然",
            "camera_height": "轻微低机位",
            "camera_angle": "轻微仰拍",
            "lens_feel": "自然偏广角",
            "subject_ratio_target": "人物占画面约38%到55%",
        })
    elif visual_goal == "更像电影镜头":
        strategy.update({
            "composition_goal": "强化叙事感与留白结构",
            "camera_angle": "正面或斜侧",
            "shot_size_target": "中景偏环境",
            "negative_space_target": "保留更多叙事留白",
            "depth_strategy": "增强前中后层次",
        })
    elif visual_goal == "逆光轮廓":
        strategy.update({
            "composition_goal": "在原有光线基础上轻微增强人物轮廓",
            "light_strategy": "仅当原图已有背光或边缘亮部时增强轮廓，不新增光源",
            "depth_strategy": "保留背景光感并保证脸部可读",
        })

    lens_pref = user_intent["lens_preference"]
    if lens_pref == "更广角冲击":
        strategy["lens_feel"] = "广角感"
        strategy["depth_strategy"] = "增强空间纵深"
        strategy["camera_angle"] = "轻微仰拍或斜侧"
        strategy["subject_ratio_target"] = "人物占画面约28%到45%"
        strategy["negative_space_target"] = "保留可读的环境留白并增强透视延展"
        strategy["composition_methods"] = strategy.get("composition_methods", "") + "；近大远小透视强化；前景锚点增强广角冲击"
    elif lens_pref == "35mm环境人像":
        strategy["lens_feel"] = "35mm环境人像感"
        strategy["subject_ratio_target"] = "人物占画面约28%到45%"
        strategy["depth_strategy"] = "保留人物与环境的尺度关系"
    elif lens_pref == "50mm标准纪实":
        strategy["lens_feel"] = "50mm标准视角"
        strategy["subject_ratio_target"] = "人物占画面约35%到55%"
    elif lens_pref == "85mm半身人像":
        strategy["lens_feel"] = "85mm半身人像感"
        strategy["shot_size_target"] = "半身或胸像"
        strategy["depth_strategy"] = "背景轻微柔化，人物边缘清晰"
    elif lens_pref == "长焦浅景深":
        strategy["lens_feel"] = "长焦浅景深感"
        strategy["depth_strategy"] = "柔化背景干扰但保持场景可读"
    elif lens_pref == "低机位广角":
        strategy["lens_feel"] = "轻微低机位广角感"
        strategy["camera_height"] = "轻微低机位"
        strategy["camera_angle"] = "轻微仰拍"
    elif lens_pref == "更突出人物":
        strategy["lens_feel"] = "中焦感"
        strategy["subject_ratio_target"] = "人物占画面约55%到72%"
    elif lens_pref == "更压缩背景":
        strategy["lens_feel"] = "长焦压缩感"
        strategy["depth_strategy"] = "压缩背景层次"
    elif lens_pref == "前景虚化层次":
        strategy["depth_strategy"] = "前景轻微虚化形成层次，脸部无遮挡"

    emotion_style = user_intent["emotion_style"]
    if emotion_style == "快乐":
        strategy["pose_strategy"] = "姿态打开，视线更积极"
    elif emotion_style == "自信":
        strategy["pose_strategy"] = "肩线打开，身体朝向稳定，眼神直接但不过度锐利"
    elif emotion_style == "松弛":
        strategy["pose_strategy"] = "姿态自然放松，减少摆拍感"
    elif emotion_style == "专注":
        strategy["pose_strategy"] = "动作克制，眼神聚焦明确"
    elif emotion_style == "清爽":
        strategy["pose_strategy"] = "表情自然轻盈，身体线条干净"
    elif emotion_style == "惆怅":
        strategy["pose_strategy"] = "视线轻微偏离镜头，动作克制"
    elif emotion_style == "神秘":
        strategy["pose_strategy"] = "保留侧向轮廓，减少正面直视"
    elif emotion_style == "温暖":
        strategy["pose_strategy"] = "表情柔和，动作自然靠近"
    elif emotion_style == "冷感":
        strategy["pose_strategy"] = "姿态稳定，表情克制"
    elif emotion_style == "高反差戏剧":
        strategy["pose_strategy"] = "姿态稳定，表情克制，利用原图已有明暗形成张力"
        strategy["light_strategy"] = "只增强原图已有明暗对比，不新增硬光源"

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
    technique_note = strategy_plan.get("technique_note", "")

    return (
        f"这是一张真实照片风格的人像优化结果。人物以{strategy_plan['shot_size_target']}呈现，"
        f"机位为{strategy_plan['camera_height']}，角度采用{strategy_plan['camera_angle']}，"
        f"画面构图目标是{strategy_plan['composition_goal']}，人物位置目标为{strategy_plan['subject_position_target']}。"
        f"镜头观感为{strategy_plan['lens_feel']}，空间层次策略为{strategy_plan['depth_strategy']}，"
        f"构图执行方法包括{strategy_plan.get('composition_methods', '三分法与留白控制')}。"
        f"摄影技法软约束：{technique_note} "
        f"情绪表达为{user_intent['emotion_style']}，用途偏向{user_intent['usage_type']}。"
        f"{light_clause} 眼神策略为{strategy_plan.get('eye_focus_strategy', '眼神自然聚焦')}。"
        f"人物状态自然上镜，优化幅度克制，不要过度执行技法；保持同一人物身份特征，环境维持原场景。"
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
    visual_goal = user_intent.get("visual_goal", "")
    lens_pref = user_intent.get("lens_preference", "")
    technique_note = strategy_plan.get("technique_note", "")
    if visual_goal == "背景更壮观":
        env_rule = "环境结构需与原图真实空间一致；可轻微增强空间层次和开阔感，但不要重建新环境。"
    elif visual_goal == "人物更突出":
        env_rule = "环境内容保持原场景真实性；可适度裁切、弱化背景干扰、提升主体占比。"
    elif visual_goal == "背景虚化分离":
        env_rule = "环境内容保持原场景真实性；只做轻微背景柔化，不要把背景抹成假景深。"
    else:
        env_rule = "环境布局、背景内容、空间结构、主要物体位置以原图为准，不要根据文字重新设计新环境。"

    lens_rule = "按所选镜头倾向执行自然透视。"
    if lens_pref == "更广角冲击":
        lens_rule = "体现轻微广角纵深即可，保留环境范围，但不要夸张拉伸脸部、四肢或空间。"
    elif lens_pref == "低机位广角":
        lens_rule = "采用轻微低机位广角感即可，比例必须自然，不要过度拉腿。"
    elif lens_pref in {"85mm半身人像", "长焦浅景深", "更压缩背景"}:
        lens_rule = "体现中长焦压缩和轻微背景柔化即可，人物眼睛和边缘必须清晰。"

    return f"""
请基于输入原图进行真实照片风格的图像编辑，生成“调整后的理想拍摄结果图”。

核心要求：
1. 保持这是同一个人物，人物身份特征、脸型、发型、眼镜、服装风格、性别与年龄感受要与原图一致；
2. {env_rule}
3. 元素锁定是最高优先级：不得新增原图不存在的人、动物、建筑、车辆、植物、家具、道具、窗户、门框、栏杆、灯具、文字、图标、前景、背景或任何装饰元素。
4. 只能对原图已经存在的元素做取景、裁切、比例、透视、明暗、色彩、清晰度和层次调整；不能为了完成风格而生成新元素。
5. 在原图基础上允许可感知但克制的摄影优化，包括适度裁切、轻微透视校正、背景层次优化和色彩统一；
6. 最终结果必须是自然、真实、写实的照片风格，不要卡通化，不要插画化，不要二次元化；
7. 不要加入箭头、图标、标签、气泡对话框、教学标注、海报标题；除“表情建议”单行文字外，不要再添加其他说明文字；
8. 不要改变人物性别，不要替换成其他陌生人，不要明显改变五官和身份特征；
9. 整体效果应像真实相机拍摄得到的优化结果，而不是绘画作品。
10. {lens_rule}
11. 可以对人物站位、机位和取景做轻到中等幅度优化，但必须严格基于原图真实内容，不得凭空造新主体，不要过度重构。
12. 人物身份与外观锁定：不得换脸，不得换人，不得改变穿着、发型、配饰，不得改变年龄、性别与体型。
13. 场景元素锁定：原图所有可见道具与背景元素必须保留；只允许可见比例与画面权重调整，不允许凭空增加、删除或替换元素。

人物处理要求：
1. 对人物脸部做自然、克制、真实的轻微美颜优化；
2. 仅允许轻微提亮肤色与轻微磨皮，保持真实毛孔与皮肤纹理，不要塑料感；
3. 可以适度优化肤质、肤色均匀度、黑眼圈，但必须保留本人真实特征，不要换脸，不要整容感，不要过度磨皮，不要失真，不要夸张修图；
4. 保留人物原有眼镜、发型、脸型和识别特征。

眼神表现要求（高优先级）：
1. 眼神必须自然有神，避免呆滞、空洞、失焦感；
2. 允许轻微增加眼部高光和对比；
3. 不改变眼睛结构比例，不做夸张放大，不做美瞳妆感；
4. 视线优先朝向镜头或镜头附近，保证人物状态精神但不过度夸张。

表情控制要求（高优先级）：
1. 保持原图表情基线与身份特征，不要出现夸张笑、露齿笑、张嘴、卡通化表情；
2. 允许轻微精神状态优化（重点提升眼神聚焦与神采、嘴角微调），但不得改变表情类型；
3. 若模型无法稳定保持自然，请优先保持原表情，不要强行改表情；
4. 严禁改变五官结构比例、嘴部闭合状态和眉眼几何关系。

光线处理要求：
1. 只能利用原图中已经存在的光线条件进行优化；
2. 不要添加原图中不存在的新光源；
3. 不要凭空增加窗光、太阳光、轮廓光、补光灯、背光灯、路灯、霓虹灯或光斑；
4. 如果原图光源方向不明显，就保持自然真实的明暗关系，不要强行制造明显的新光线结构。

构图执行要求：
1. 优先执行当前选择对应的一个主摄影技法；三分法、引导线、留白、前景层次、框架、对角线等只在原图已有相应视觉元素时使用；
2. 如果原图没有框架、前景、引导线或逆光，就不要为了实现该技法而添加门窗、树叶、栏杆、灯光或其他新物体；
3. 对于主体线条和空间线条，优先保证水平垂直关系自然，避免歪斜畸变；
4. 保留真实摄影质感，不要做海报化夸张特效；
5. 可通过色彩对比和明度层级突出主体，但不得破坏真实肤色；
6. 构图优化要可感知但克制，避免炫技式过度重构。

画面文字要求：
1. 不要在生成图中添加任何文字；
2. 表情建议将由前端叠加显示，不属于生成图内容。

人物性别：
{ai_result.get("subject_gender", "")}

用户选择目标：
- 视觉目标：{user_intent.get("visual_goal", "")}
- 情绪风格：{user_intent.get("emotion_style", "")}
- 拍摄用途：{user_intent.get("usage_type", "")}
- 镜头倾向：{user_intent.get("lens_preference", "")}

专业摄影策略：
- 摄影技法软约束：{technique_note}
- 构图目标：{strategy_plan.get("composition_goal", "")}
- 机位高度：{strategy_plan.get("camera_height", "")}
- 拍摄角度：{strategy_plan.get("camera_angle", "")}
- 景别目标：{strategy_plan.get("shot_size_target", "")}
- 镜头观感：{strategy_plan.get("lens_feel", "")}
- 人物位置目标：{strategy_plan.get("subject_position_target", "")}
- 人物占比目标：{strategy_plan.get("subject_ratio_target", "")}
- 留白策略：{strategy_plan.get("negative_space_target", "")}
- 空间策略：{strategy_plan.get("depth_strategy", "")}
- 光线策略：{strategy_plan.get("light_strategy", "")}
- 姿态策略：{strategy_plan.get("pose_strategy", "")}
- 眼神策略：{strategy_plan.get("eye_focus_strategy", "")}
- 构图方法：{strategy_plan.get("composition_methods", "")}
- 色彩策略：{strategy_plan.get("color_strategy", "")}
- 几何策略：{strategy_plan.get("geometry_strategy", "")}

目标结果描述：
{targeted_ideal_prompt}

理想拍摄结果描述：
{ai_result.get("ideal_image_prompt", "")}

光线判断参考：
{ai_result.get("light_source_inference", "")}

反向限制：
- 不要卡通风格
- 不要插画风格
- 不要动漫风格
- 不要二次元
- 不要气泡对话框
- 不要对白框
- 不要大段文字
- 不要任何叠加文字
- 不要遮挡主体的文字
- 不要教学标注
- 不要替换人物
- 不要错误性别
- 不要改变人物穿着
- 不要改变人物发型和配饰
- 不要凭空新增或删除场景关键元素
- 不要新增原图不存在的人、物、建筑、植物、车辆、道具、灯具、门窗、前景或背景
- 不要新增原图不存在的光源
- 不要夸张磨皮
- 不要整容脸
- 不要网红滤镜感
""".strip()


# =========================
# 图像编辑
# =========================
def edit_with_qwen_image_edit(image_path: str, prompt_text: str) -> str:
    image_data_url = image_file_to_data_url(image_path)

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "qwen-image-edit-max",
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
        "parameters": {
            "n": 1,
            "negative_prompt": "微笑,大笑,露齿,张嘴,改变表情,表情变形,卡通风格,插画风格,动漫风格,二次元,气泡对话框,对白框,教学标注,箭头,标签,海报标题,大段文字,遮挡主体的文字,陌生人,新增人物,新增物体,新增道具,新增家具,新增建筑,新增植物,新增车辆,新增前景,新增背景,新增灯具,新增光源,新窗户,新门框,栏杆,替换背景,删除背景元素,性别错误,明显换脸,化妆,眼影,口红,美妆,脂粉气,浓妆,烟熏妆,条纹噪点,网纹,摩尔纹,扫描线,网格噪声,纹理污染,呆滞眼神,空洞眼神,失焦眼神,斗鸡眼",
            "prompt_extend": False,
            "watermark": False,
            "size": "1024*1536",
        }
    }

    resp = requests.post(MULTIMODAL_GEN_URL, headers=headers, json=payload, timeout=180)

    print("=== qwen-image-edit-max 状态码 ===")
    print(resp.status_code)
    print(resp.text[:3000])

    if resp.status_code != 200:
        raise requests.HTTPError(response=resp)

    return parse_qwen_image_edit_url(resp.json())


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
            # 强制风格：写实、非卡通 (使用自然语言否定)
            final_prompt = f"photorealistic, real photo, 8k, raw photo, realistic texture. {prompt_text}. Do not use cartoon style. Do not use anime style. Do not use 3d render style. Do not use painting style. Do not use illustration style."
            
            payload = {
                "model": model_name,
                "prompt": final_prompt,
                "image": image_data_url,
                "size": "1024x1536",
                "n": 1,
            }
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
                print(f"=== Doubao-Seedream-5.0-lite 请求异常: {endpoint} ===")
                print(str(e))
                continue
            print(f"=== Doubao-Seedream-5.0-lite 状态码: {endpoint} | model={model_name} ===")
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


def check_image_model_ready(model_name: str) -> Tuple[bool, str]:
    if model_name == "qwen-image-edit-max":
        return bool(DASHSCOPE_API_KEY), "未配置 DASHSCOPE_API_KEY"
    if model_name == "Doubao-Seedream-5.0-lite":
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
    model_to_runner = {
        "qwen-image-edit-max": edit_with_qwen_image_edit,
        "Doubao-Seedream-5.0-lite": edit_with_doubao_seedream,
        "jimeng-ai-image-4.0": edit_with_jimeng_image,
    }

    active_models = get_ready_models(MODEL_COMPARE_LIST)
    for model_name in active_models:
        runner = model_to_runner.get(model_name)
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
            if chosen is None or float(item.get("quality_score", -999)) > float(chosen.get("quality_score", -999)):
                chosen = item
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

    if candidates:
        qwen_item = next((x for x in candidates if x.get("model") == "qwen-image-edit-max" and not x.get("error")), None)
        if qwen_item:
            qwen_reason = str(qwen_item.get("quality_reason", ""))
            qwen_score = float(qwen_item.get("quality_score", -999.0))
            if "条纹伪影" not in qwen_reason and qwen_score > -3.0:
                chosen = qwen_item

        if chosen is None:
            valid_items = []
            for x in candidates:
                if x.get("error"):
                    continue
                reason = str(x.get("quality_reason", ""))
                score = float(x.get("quality_score", -999.0))
                if "条纹伪影" in reason or score <= -3.0:
                    continue
                valid_items.append(x)
            if valid_items:
                chosen = max(valid_items, key=lambda z: float(z.get("quality_score", -999.0)))

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
    model_to_runner = {
        "qwen-image-edit-max": edit_with_qwen_image_edit,
        "Doubao-Seedream-5.0-lite": edit_with_doubao_seedream,
        "jimeng-ai-image-4.0": edit_with_jimeng_image,
    }
    errors: List[str] = []
    for model_name in get_ready_models(IMAGE_EDIT_MODELS):
        runner = model_to_runner.get(model_name)
        if runner is None:
            continue
        try:
            return runner(image_path, prompt_text), model_name, ""
        except Exception as e:
            err = f"{model_name} 失败：{e}"
            errors.append(err)
            print("[图像编辑失败]", err)
    if not errors:
        return "", "", "未配置任何可用图像模型，请先配置 Qwen/豆包/智谱/即梦 的密钥。"
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
