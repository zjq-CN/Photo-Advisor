"""图像质量评估（run_model_compare 依赖；单列以避免 vision↔imagegen 循环）。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import os
import requests
from app.config import UPLOAD_FOLDER



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
