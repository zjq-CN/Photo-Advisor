"""混元图生图适配器（含 SDK 依赖与 HUNYUAN_* 全局状态）。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
import base64
import json
import re
import requests
import time
from app.config import load_env
from app.parsing import parse_image_base64_any, parse_image_url_any
from app.storage import save_base64_image_to_uploads


try:
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.hunyuan.v20230901 import hunyuan_client, models as hunyuan_models
    TENCENT_SDK_AVAILABLE = True
except ImportError:
    TENCENT_SDK_AVAILABLE = False

HUNYUAN_API_KEY = load_env("HUNYUAN_API_KEY")
HUNYUAN_SECRET_ID = load_env("HUNYUAN_SECRET_ID") # For SDK
HUNYUAN_SECRET_KEY = load_env("HUNYUAN_SECRET_KEY") # For SDK


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
