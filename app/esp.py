"""ESP32 屏幕 JPEG 构建与推送。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import os
import requests
from app.config import UPLOAD_FOLDER, get_runtime_esp_config, parse_int



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


def push_queue(
    queue: List[str],
    stop_on_first_failure: bool = False,
    add_skip_note: bool = False,
) -> Tuple[bool, str, int, int]:
    """逐张推送队列。

    阶段③ 把 routes/api.py 里两处重复循环收成一处实现，两个开关
    分别还原两侧原有语义（分支条件与原实现逐字一致）：
    - stop_on_first_failure=True  （生成路径）：首张失败即 break，
      并在消息尾部注记"其余 N 张已跳过"；
    - stop_on_first_failure=False（手动推送路径）：全部尝试，不加注记。
    """
    success_count = 0
    attempted = 0
    details: List[str] = []
    for i, ref in enumerate(queue):
        ok, msg = push_image_to_esp(ref, append=i > 0)
        attempted += 1
        details.append(msg)
        if ok:
            success_count += 1
        elif stop_on_first_failure:
            break

    total = len(queue)
    tail = details[-1] if details else ""
    if add_skip_note:
        skipped = total - attempted
        note = f"（其余 {skipped} 张已跳过）" if skipped > 0 else ""
        message = f"ESP push {success_count}/{total}. {tail}{note}"
    else:
        message = f"ESP push {success_count}/{total}. {tail}".strip()
    return success_count > 0, message, success_count, total


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
