"""sessions/latest_result/uploads 读写。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
from typing import Any, Callable, Dict, List, Optional, Tuple
import base64
from datetime import datetime
import json
import os
import re
import uuid
from app.config import ALLOWED_EXTENSIONS, LATEST_RESULT_JSON, SESSION_FOLDER, UPLOAD_FOLDER



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
