"""视觉理解模型适配与选路。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
import app.imagegen.hunyuan as _hunyuan_state
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import requests
from app.config import COMPAT_CHAT_URL, DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, DEEPSEEK_CHAT_URL, DEEPSEEK_VISION_MODEL, HUNYUAN_CHAT_URL, HUNYUAN_VISION_MODEL, ZHIPU_API_KEY, ZHIPU_CHAT_URL, ZHIPU_REASONING_EFFORT, ZHIPU_VISION_MODEL
from app.contracts import VISION_MODE_PRESETS
from app.parsing import extract_json_from_text, is_busy_or_rate_limited, normalize_ai_result
from app.prompts import build_analysis_prompt
from app.storage import image_file_to_data_url



# =========================
# 视觉理解
# =========================
def call_vision_once(model_name: str, image_path: str, geometry_block: str = "", extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    extra_params = extra_params or {}
    if model_name.startswith("hunyuan") or model_name.startswith("hy-"):
        return call_vision_hunyuan_once(model_name, image_path, geometry_block)
    if model_name.startswith("deepseek"):
        return call_vision_deepseek_once(model_name, image_path, geometry_block, extra_params)
    if model_name.startswith("glm-"):
        return call_vision_zhipu_once(model_name, image_path, geometry_block, extra_params)
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
        # enable_thinking 是百炼私有参数：OpenAI SDK 里经 extra_body 合并进 JSON 顶层，裸 requests 必须直接放顶层
        # （历史坑：曾把 "extra_body" 当 JSON 字段发送，服务端忽略 → 思考从未关闭，实测 39.5s→5.8s）
        "enable_thinking": bool(extra_params.get("enable_thinking", False)),
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
    if _hunyuan_state.HUNYUAN_API_KEY and _hunyuan_state.HUNYUAN_API_KEY.startswith("AKID"):
        print("!!! 警告: HUNYUAN_API_KEY 看起来像是腾讯云 SecretId (以 AKID 开头)。")
        print("!!! 混元 OpenAI 兼容接口需要单独的 API Key，而非 SecretId/SecretKey。")
        print("!!! 请前往 https://console.cloud.tencent.com/hunyuan/start 获取 API Key。")

    image_data_url = image_file_to_data_url(image_path)
    prompt = build_analysis_prompt(geometry_block)
    headers = {
        "Authorization": f"Bearer {_hunyuan_state.HUNYUAN_API_KEY}",
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


def call_vision_deepseek_once(model_name: str, image_path: str, geometry_block: str = "", extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
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
        # 思考档（用户拍板保留思考提升兜底质量）：v4 系默认即 enabled + effort=high，显式声明防上游默认变化
        # 注意：thinking 开启时勿配 reasoning_effort=low + 小 max_tokens——实测思考恰好吃满预算导致 content 截断
        # V4.1 Flash reasoning 更 verbose，8192 才够 reasoning + content 各自充分
        "thinking": (extra_params or {}).get("thinking", {"type": "enabled"}),
        "max_tokens": 8192,
    }
    # V4.1 Flash 偶尔在 JSON 字符串值内混入未转义双引号，重试一次即可恢复
    last_err: Optional[Exception] = None
    for _attempt in range(2):
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
        try:
            result = extract_json_from_text(content_text)
            return normalize_ai_result(result)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            print(f"[retry] JSON 解析失败，重试: {e}")
    raise last_err  # type: ignore[misc]


def call_vision_zhipu_once(model_name: str, image_path: str, geometry_block: str = "", extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
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
        "max_tokens": 4096,  # glm-5.3-flash 为思考型：reasoning+答案共享配额，2048 会被思考吃光导致 content 为空
    }
    if model_name.startswith("glm-5.3"):
        payload["reasoning_effort"] = (extra_params or {}).get("reasoning_effort") or ZHIPU_REASONING_EFFORT
    resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=180)  # 思考型模型单次可达 ~120s，120s 会误杀
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
        return bool(_hunyuan_state.HUNYUAN_API_KEY), "未配置 HUNYUAN_API_KEY"
    if model_name.startswith("deepseek"):
        return bool(DEEPSEEK_API_KEY), "未配置 DEEPSEEK_API_KEY"
    if model_name.startswith("glm-"):
        return bool(ZHIPU_API_KEY), "未配置 ZHIPU_API_KEY"
    return bool(DASHSCOPE_API_KEY), "未配置 DASHSCOPE_API_KEY"


def call_vision_auto(image_path: str, geometry_block: str = "", mode: str = "standard") -> Tuple[Dict[str, str], str]:
    preset = VISION_MODE_PRESETS.get(mode) or VISION_MODE_PRESETS["standard"]
    params_map = preset["params"]
    last_error: Optional[Exception] = None
    ready_models: List[str] = []
    for model_name in preset["chain"]:
        ready, _ = check_vision_model_ready(model_name)
        if not ready:
            continue
        ready_models.append(model_name)
        try:
            result = call_vision_once(model_name, image_path, geometry_block, params_map.get(model_name))
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
