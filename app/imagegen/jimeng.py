"""即梦图生图适配器（含火山签名/轮询）。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
from typing import Any, Callable, Dict, List, Optional, Tuple
import base64
import cv2
from datetime import datetime
import hashlib
import hmac
import json
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
import requests
import time
from app.config import JIMENG_ACCESS_KEY_ID, JIMENG_IMAGE_URL, JIMENG_REQ_KEY, JIMENG_SECRET_ACCESS_KEY
from app.parsing import parse_image_base64_any, parse_image_url_any, parse_task_id_any, safe_resp_text
from app.storage import save_base64_image_to_uploads



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
