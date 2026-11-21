"""豆包图生图适配器。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
from typing import Any, Callable, Dict, List, Optional, Tuple
import requests
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from app.config import DOUBAO_API_KEY, DOUBAO_ENDPOINT_ID, DOUBAO_IMAGE_FALLBACK_MODEL, DOUBAO_IMAGE_MODEL, DOUBAO_IMAGE_URL, DOUBAO_MODEL_LIST_URL
from app.parsing import parse_image_base64_any, parse_image_url_any, safe_resp_text
from app.storage import image_file_to_data_url, save_base64_image_to_uploads



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
