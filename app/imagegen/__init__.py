"""图生图编排：注册表/就绪检查/并发对比/自动选择。"""

# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。
import app.imagegen.hunyuan as _hunyuan_state
from typing import Any, Callable, Dict, List, Optional, Tuple
import requests
import threading
from app.config import DASHSCOPE_API_KEY, DOUBAO_API_KEY, JIMENG_ACCESS_KEY_ID, JIMENG_IMAGE_URL, JIMENG_SECRET_ACCESS_KEY, ZHIPU_API_KEY
from app.contracts import IMAGE_EDIT_MODELS, MODEL_COMPARE_LIST, MODEL_QWEN_EDIT_MAX, MODEL_QWEN_IMAGE_20, MODEL_SEEDREAM_50, MODEL_WAN_27
from app.parsing import safe_resp_text
from app.imagegen.dashscope import edit_with_qwen_image_20, edit_with_qwen_image_edit, edit_with_wan27_image
from app.imagegen.doubao import edit_with_doubao_seedream
from app.quality import evaluate_image_quality



def get_image_model_runner(model_name: str):
    return {
        MODEL_WAN_27: edit_with_wan27_image,
        MODEL_QWEN_IMAGE_20: edit_with_qwen_image_20,
        MODEL_SEEDREAM_50: edit_with_doubao_seedream,
        MODEL_QWEN_EDIT_MAX: edit_with_qwen_image_edit,
    }.get(model_name)


def check_image_model_ready(model_name: str) -> Tuple[bool, str]:
    if model_name in {MODEL_WAN_27, MODEL_QWEN_IMAGE_20, MODEL_QWEN_EDIT_MAX}:
        return bool(DASHSCOPE_API_KEY), "未配置 DASHSCOPE_API_KEY"
    if model_name == MODEL_SEEDREAM_50:
        return bool(DOUBAO_API_KEY), "未配置 DOUBAO_API_KEY"
    if model_name == "zhipu-glm-image":
        return bool(ZHIPU_API_KEY), "未配置 ZHIPU_API_KEY"
    if model_name == "jimeng-ai-image-4.0":
        if not JIMENG_IMAGE_URL:
            return False, "未配置 JIMENG_IMAGE_URL"
        if not JIMENG_ACCESS_KEY_ID or not JIMENG_SECRET_ACCESS_KEY:
            return False, "未配置即梦 AccessKey"
        return True, ""
    if model_name == "hunyuan-dit":
        return bool(_hunyuan_state.HUNYUAN_API_KEY) or (bool(_hunyuan_state.HUNYUAN_SECRET_ID) and bool(_hunyuan_state.HUNYUAN_SECRET_KEY)), "未配置 HUNYUAN_API_KEY 或 SecretId/Key"
    return False, f"未知模型：{model_name}"


def get_ready_models(model_names: List[str]) -> List[str]:
    ready: List[str] = []
    for model_name in model_names:
        ok, _ = check_image_model_ready(model_name)
        if ok:
            ready.append(model_name)
    return ready


def _valid_image_url(url: str) -> bool:
    s = str(url).strip()
    return bool(s) and (
        s.startswith("http://")
        or s.startswith("https://")
        or s.startswith("/uploads/")
    )


def _attempt_model(
    model_name: str,
    image_path: str,
    prompt_text: str,
    generation_settings: Optional[Dict[str, Any]],
    user_intent: Optional[Dict[str, str]],
    mark_first_ok: Callable[[str], None],
) -> Optional[Dict[str, Any]]:
    """单个模型的一次尝试；runner 不存在时返回 None。

    mark_first_ok 在 URL 校验通过后、质量打分之前调用——与原实现的标记时机同序。
    """
    runner = get_image_model_runner(model_name)
    if runner is None:
        return None
    try:
        raw_url = runner(image_path, prompt_text, generation_settings)
        if not _valid_image_url(raw_url):
            raise RuntimeError("模型返回空图片地址。")
        # 第一个成功者即标记为主图模型，不阻塞等待质量打分
        mark_first_ok(model_name)
        quality_score, quality_reason = evaluate_image_quality(
            model_name,
            raw_url,
            generation_settings=generation_settings,
            user_intent=user_intent,
        )
        return {
            "model": model_name,
            "raw_diagram_url": raw_url,
            "diagram_url": raw_url,
            "beauty_applied": False,
            "error": "",
            "quality_score": quality_score,
            "quality_reason": quality_reason,
        }
    except Exception as e:
        err = str(e)
        if not err and isinstance(e, requests.HTTPError):
            try:
                err = safe_resp_text(e.response)
            except Exception:
                err = ""
        if "InternalError" in err or "submit algo service error" in err:
            err = "服务繁忙（InternalError），已自动回退"
        return {
            "model": model_name,
            "raw_diagram_url": "",
            "diagram_url": "",
            "beauty_applied": False,
            "error": err,
            "quality_score": -999.0,
            "quality_reason": "",
        }


def run_model_compare(
    image_path: str,
    prompt_text: str,
    generation_settings: Optional[Dict[str, Any]] = None,
    user_intent: Optional[Dict[str, str]] = None,
    on_candidate: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """并发调用多个生图模型，谁先成功谁作为主图（不用等全部跑完再选优）。

    返回结构保持兼容：diagram_url/image_model 为主图结果，
    model_candidates 为全部候选（按 MODEL_COMPARE_LIST 顺序），供前端对比展示。

    on_candidate: 每完成一个模型（成功或失败）就调用一次 (model_name, item)，
    用于增量通知（例如把已完成的候选实时写回 latest.json，前端逐张显示）。
    """
    active_models = get_ready_models(MODEL_COMPARE_LIST)
    if not active_models:
        return _empty_compare_result()

    results: Dict[str, Dict[str, Any]] = {}
    first_ok_model: List[str] = []
    lock = threading.Lock()
    first_ok_lock = threading.Lock()

    def _mark_first_ok(model_name: str) -> None:
        with first_ok_lock:
            if not first_ok_model:
                first_ok_model.append(model_name)

    def _worker(model_name: str) -> None:
        item = _attempt_model(model_name, image_path, prompt_text,
                              generation_settings, user_intent, _mark_first_ok)
        if item is None:
            return
        with lock:
            results[model_name] = item
        if on_candidate is not None:
            try:
                on_candidate(model_name, item)
            except Exception:
                pass

    threads = [
        threading.Thread(target=_worker, args=(m,), daemon=True)
        for m in active_models
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return _select_compare_result(active_models, results, first_ok_model)


def _empty_compare_result() -> Dict[str, Any]:
    return {
        "diagram_url": "",
        "raw_diagram_url": "",
        "image_model": "",
        "diagram_error": "未配置任何可用图像模型。",
        "beauty_applied": False,
        "model_candidates": [],
    }


def _select_compare_result(
    active_models: List[str],
    results: Dict[str, Dict[str, Any]],
    first_ok_model: List[str],
) -> Dict[str, Any]:
    """按原实现规则挑主图：首个成功者优先，失败则回退到任一成功候选。"""
    candidates = [results[m] for m in active_models if m in results]
    chosen_model = first_ok_model[0] if first_ok_model else ""
    chosen = results.get(chosen_model) if chosen_model else None

    # 若首个成功模型因质量检测被标记失败（极少见），回退到任一成功候选
    if chosen is None or chosen.get("error"):
        chosen = next((c for c in candidates if not c.get("error")), None)

    if chosen:
        return {
            "diagram_url": chosen["diagram_url"],
            "raw_diagram_url": chosen["raw_diagram_url"],
            "image_model": chosen["model"],
            "diagram_error": "",
            "beauty_applied": chosen["beauty_applied"],
            "model_candidates": candidates,
        }

    errors = [c.get("error", "") for c in candidates if c.get("error")]
    return {
        "diagram_url": "",
        "raw_diagram_url": "",
        "image_model": "",
        "diagram_error": "；".join(errors) or "所有生图模型调用失败。",
        "beauty_applied": False,
        "model_candidates": candidates,
    }


def edit_image_auto(
    image_path: str,
    prompt_text: str,
    generation_settings: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, str]:
    errors: List[str] = []
    for model_name in get_ready_models(IMAGE_EDIT_MODELS):
        runner = get_image_model_runner(model_name)
        if runner is None:
            continue
        try:
            return runner(image_path, prompt_text, generation_settings), model_name, ""
        except Exception as e:
            err = f"{model_name} 失败：{e}"
            errors.append(err)
            print("[图像编辑失败]", err)
    if not errors:
        return "", "", "未配置任何可用图像模型，请先配置 DASHSCOPE_API_KEY 或 DOUBAO_API_KEY。"
    return "", "", "；".join(errors)
