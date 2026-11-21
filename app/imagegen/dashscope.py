"""阿里百炼图生图适配器。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
from typing import Any, Callable, Dict, List, Optional, Tuple
import requests
from app.config import DASHSCOPE_API_KEY, MULTIMODAL_GEN_URL, QWEN_IMAGE_EDIT_MAX_MODEL, QWEN_IMAGE_MODEL, WAN_IMAGE_MODEL
from app.parsing import parse_image_url_any
from app.prompts import QWEN_IMAGE_NEGATIVE_PROMPT
from app.storage import image_file_to_data_url



def edit_with_dashscope_image_model(
    image_path: str,
    prompt_text: str,
    model_id: str,
    use_negative_prompt: bool,
    generation_settings: Optional[Dict[str, Any]] = None,
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
    }
    requested_size = (generation_settings or {}).get("dashscope_size")
    if requested_size:
        parameters["size"] = requested_size
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


def edit_with_wan27_image(
    image_path: str,
    prompt_text: str,
    generation_settings: Optional[Dict[str, Any]] = None,
) -> str:
    return edit_with_dashscope_image_model(
        image_path,
        prompt_text,
        WAN_IMAGE_MODEL,
        use_negative_prompt=False,
        generation_settings=generation_settings,
    )


def edit_with_qwen_image_20(
    image_path: str,
    prompt_text: str,
    generation_settings: Optional[Dict[str, Any]] = None,
) -> str:
    return edit_with_dashscope_image_model(
        image_path,
        prompt_text,
        QWEN_IMAGE_MODEL,
        use_negative_prompt=True,
        generation_settings=generation_settings,
    )


def edit_with_qwen_image_edit(
    image_path: str,
    prompt_text: str,
    generation_settings: Optional[Dict[str, Any]] = None,
) -> str:
    return edit_with_dashscope_image_model(
        image_path,
        prompt_text,
        QWEN_IMAGE_EDIT_MAX_MODEL,
        use_negative_prompt=True,
        generation_settings=generation_settings,
    )
