"""智谱图生图适配器。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
import requests
from app.config import ZHIPU_API_KEY, ZHIPU_IMAGE_MODEL, ZHIPU_IMAGE_URL
from app.parsing import parse_image_url_any
from app.storage import image_file_to_data_url



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
