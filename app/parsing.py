"""模型返回体解析：JSON 修复与提取、URL/base64/task_id 提取、AI 结果归一化。

阶段③ 从 contracts.py 拆出（P0 单文件 ≤400 行）。函数体逐字节未改。
"""

from typing import Any, Dict, Optional
import json
import re
import requests


def _repair_json_string(raw: str) -> str:
    """修复模型输出中常见的 JSON 格式错误（如中文引号""、字符串内未转义的双引号）。"""
    # 1) 中文引号 → 转义双引号
    raw = raw.replace('\u201c', '\\"').replace('\u201d', '\\"')
    # 2) 字符串值内的未转义双引号：逐字符扫描，只处理 JSON key 之外的位置
    out, in_str, i = [], False, 0
    while i < len(raw):
        ch = raw[i]
        if ch == '\\' and in_str:
            out.append(ch)
            if i + 1 < len(raw):
                out.append(raw[i + 1])
            i += 2
            continue
        if ch == '"':
            if not in_str:
                in_str = True
                out.append(ch)
            else:
                # 判断这个 " 是值结束符还是值内部的引号
                nxt = next((raw[j] for j in range(i + 1, len(raw)) if raw[j] not in ' \t\r\n'), '')
                if nxt in (',', '}', ']'):
                    in_str = False
                    out.append(ch)
                else:
                    out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def extract_json_from_text(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    brace = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(1))
        except Exception:
            # 最后尝试修复格式错误
            try:
                return json.loads(_repair_json_string(brace.group(1)))
            except Exception:
                pass

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


def normalize_ai_result(result: Dict[str, Any]) -> Dict[str, str]:
    keys = [
        "scene_summary",
        "main_issue",
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
