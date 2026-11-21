"""图像生成流程（阶段③ 从 routes/api.py 的 api_generate 拆出）。

拆法：按语义段落抽函数 + 字典构造收为小工厂。所有 if 判断与字典键集
逐字保留，未合并、未改写分支条件。
"""

from datetime import datetime
import os
from typing import Any, Dict

from flask import jsonify

from app.config import parse_bool
from app.contracts import (
    MODEL_COMPARE_ENABLED_DEFAULT,
    MODEL_COMPARE_LIST,
    normalize_user_intent,
)
from app.esp import build_esp_push_queue, push_queue
from app.imagegen import edit_image_auto, run_model_compare
from app.prompts import build_edit_prompt, build_targeted_ideal_prompt
from app.storage import load_session_data, save_latest_result, validate_session_id
from app.strategy import build_generation_settings, build_guidance_text, build_strategy_plan


def prepare_generate_request(payload: Dict[str, Any]):
    """校验并载入生成上下文。

    返回 (err, ctx)：err 非 None 时由路由原样 return（响应体与改前逐字相同）；
    否则 ctx 为后续各步所需的全部输入。
    """
    if payload.get("generation_confirmed") is not True:
        return (jsonify({
            "ok": False,
            "msg": "未收到生成按钮的明确确认，已拒绝启动图像生成。",
        }), 400), None

    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        return (jsonify({"ok": False, "msg": "缺少 session_id"}), 400), None
    try:
        session_id = validate_session_id(session_id)
    except ValueError as e:
        return (jsonify({"ok": False, "msg": str(e)}), 400), None

    session_data = load_session_data(session_id)
    image_path = session_data.get("image_path", "")
    if not image_path or not os.path.exists(image_path):
        return (jsonify({"ok": False, "msg": "会话原图不存在，请重新拍照。"}), 400), None

    diagnosis_report = session_data.get("diagnosis_report", {})
    if not isinstance(diagnosis_report, dict):
        diagnosis_report = {}

    user_intent = normalize_user_intent(payload.get("user_intent", {}))
    generation_settings = build_generation_settings(user_intent)
    debug_compare = parse_bool(
        str(payload.get("debug_compare", MODEL_COMPARE_ENABLED_DEFAULT)),
        MODEL_COMPARE_ENABLED_DEFAULT,
    )
    strategy_plan = build_strategy_plan(diagnosis_report, user_intent)
    targeted_ideal_prompt = build_targeted_ideal_prompt(diagnosis_report, strategy_plan, user_intent)
    edit_prompt = build_edit_prompt(diagnosis_report, strategy_plan, user_intent, targeted_ideal_prompt)
    final_guidance = build_guidance_text(diagnosis_report, strategy_plan, user_intent)
    final_guidance["ideal_image_prompt"] = targeted_ideal_prompt

    return None, {
        "session_id": session_id,
        "session_data": session_data,
        "image_path": image_path,
        "diagnosis_report": diagnosis_report,
        "user_intent": user_intent,
        "generation_settings": generation_settings,
        "debug_compare": debug_compare,
        "strategy_plan": strategy_plan,
        "targeted_ideal_prompt": targeted_ideal_prompt,
        "edit_prompt": edit_prompt,
        "final_guidance": final_guidance,
    }


def build_generating_payload(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """生成中的快照：让 display 页立即展示诊断信息。"""
    session_data = ctx["session_data"]
    return {
        "ok": True,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": ctx["session_id"],
        "stage": "generating",
        "filename": session_data.get("filename", ""),
        "preview_url": session_data.get("preview_url", ""),
        "vision_model": session_data.get("vision_model", ""),
        "image_model": "",
        "diagnosis_report": ctx["diagnosis_report"],
        "user_intent": ctx["user_intent"],
        "generation_settings": ctx["generation_settings"],
        "strategy_plan": ctx["strategy_plan"],
        "ai_result": ctx["final_guidance"],
        "edit_prompt": ctx["edit_prompt"],
        "raw_diagram_url": "",
        "diagram_url": "",
        "diagram_error": "",
        "beauty_applied": False,
        "beauty_enabled": False,
        "model_candidates": [],
        "esp_push_ok": False,
        "esp_push_msg": "",
        "esp_pushed_image": "",
        "esp_push_count": 0,
        "esp_push_total": 0,
    }


def _run_compare(ctx: Dict[str, Any], generating_payload: Dict[str, Any]) -> Dict[str, Any]:
    """多模型对比分支（原 debug_compare=True 的整段，逐字搬动）。"""
    # 每完成一个模型，就把已完成候选增量写回，前端逐张显示，不用等全部完成
    partial_results: Dict[str, Dict[str, Any]] = {}

    def on_candidate(model_name: str, item: Dict[str, Any]) -> None:
        partial_results[model_name] = item
        ordered = [partial_results[m] for m in MODEL_COMPARE_LIST if m in partial_results]
        snapshot = dict(generating_payload)
        snapshot["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snapshot["model_candidates"] = ordered
        first_ok = next((c for c in ordered if not c.get("error")), None)
        if first_ok:
            snapshot["diagram_url"] = first_ok.get("diagram_url", "")
            snapshot["image_model"] = first_ok.get("model", "")
        save_latest_result(snapshot)

    compare_data = run_model_compare(
        ctx["image_path"],
        ctx["edit_prompt"],
        generation_settings=ctx["generation_settings"],
        user_intent=ctx["user_intent"],
        on_candidate=on_candidate,
    )
    return {
        "diagram_url": compare_data.get("diagram_url", ""),
        "image_model": compare_data.get("image_model", ""),
        "diagram_error": compare_data.get("diagram_error", ""),
        "raw_diagram_url": compare_data.get("raw_diagram_url",
                                            compare_data.get("diagram_url", "")),
        "beauty_applied": bool(compare_data.get("beauty_applied", False)),
        "model_candidates": compare_data.get("model_candidates", []),
    }


def run_generation_models(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """跑生图（对比分支或单模型分支），返回统一的四个结果字段。"""
    if ctx["debug_compare"]:
        generating_payload = build_generating_payload(ctx)
        save_latest_result(generating_payload)
        return _run_compare(ctx, generating_payload)

    diagram_url, used_image_model, diagram_error = edit_image_auto(
        ctx["image_path"],
        ctx["edit_prompt"],
        generation_settings=ctx["generation_settings"],
    )
    raw_diagram_url = diagram_url
    beauty_applied = False
    model_candidates = [{
        "model": used_image_model,
        "raw_diagram_url": raw_diagram_url,
        "diagram_url": diagram_url,
        "beauty_applied": beauty_applied,
        "error": diagram_error if not diagram_url else "",
    }]
    return {
        "diagram_url": diagram_url,
        "image_model": used_image_model,
        "diagram_error": diagram_error,
        "raw_diagram_url": raw_diagram_url,
        "beauty_applied": beauty_applied,
        "model_candidates": model_candidates,
    }


def push_generated(ctx: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """仅推送 AI 生成图；多图队列供屏幕端滑动查看。"""
    # 仅推送 AI 生成图，不再回退推送原始拍照图。
    esp_target_image = str(result["diagram_url"] or "").strip()
    # 自动流程推送多图队列（主图 + 其余候选图），供屏幕端滑动查看。
    esp_queue = build_esp_push_queue(esp_target_image, result["model_candidates"], include_all=True)
    if not esp_queue:
        esp_push_ok, esp_push_msg = False, "No image to push."
        esp_push_count = 0
        esp_push_total = 0
    else:
        esp_push_ok, esp_push_msg, esp_push_count, esp_push_total = push_queue(
            esp_queue, stop_on_first_failure=True, add_skip_note=True)
    print(f"[ESP] {esp_push_msg}")
    return {
        "esp_target_image": esp_target_image,
        "esp_push_ok": esp_push_ok,
        "esp_push_msg": esp_push_msg,
        "esp_push_count": esp_push_count,
        "esp_push_total": esp_push_total,
    }


def apply_to_session(ctx: Dict[str, Any], result: Dict[str, Any], esp: Dict[str, Any]) -> None:
    """把生成结果写回会话（原地改 session_data，与改前一致）。"""
    session_data = ctx["session_data"]
    session_data["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_data["stage"] = "generated"
    session_data["user_intent"] = ctx["user_intent"]
    session_data["generation_settings"] = ctx["generation_settings"]
    session_data["strategy_plan"] = ctx["strategy_plan"]
    session_data["edit_prompt"] = ctx["edit_prompt"]
    session_data["raw_diagram_url"] = result["raw_diagram_url"]
    session_data["diagram_url"] = result["diagram_url"]
    session_data["diagram_error"] = result["diagram_error"]
    session_data["image_model"] = result["image_model"]
    session_data["beauty_applied"] = result["beauty_applied"]
    session_data["beauty_enabled"] = False
    session_data["model_candidates"] = result["model_candidates"]
    session_data["ai_result"] = ctx["final_guidance"]
    session_data["esp_push_ok"] = esp["esp_push_ok"]
    session_data["esp_push_msg"] = esp["esp_push_msg"]
    session_data["esp_pushed_image"] = esp["esp_target_image"] if esp["esp_push_ok"] else ""
    session_data["esp_push_count"] = esp["esp_push_count"]
    session_data["esp_push_total"] = esp["esp_push_total"]


def build_latest_payload(ctx: Dict[str, Any], result: Dict[str, Any], esp: Dict[str, Any]) -> Dict[str, Any]:
    """写回 latest_result.json 的快照（键集与改前逐字相同）。"""
    session_data = ctx["session_data"]
    return {
        "ok": True,
        "time": session_data["time"],
        "session_id": ctx["session_id"],
        "stage": "generated",
        "filename": session_data.get("filename", ""),
        "preview_url": session_data.get("preview_url", ""),
        "vision_model": session_data.get("vision_model", ""),
        "image_model": result["image_model"],
        "diagnosis_report": ctx["diagnosis_report"],
        "user_intent": ctx["user_intent"],
        "generation_settings": ctx["generation_settings"],
        "strategy_plan": ctx["strategy_plan"],
        "ai_result": ctx["final_guidance"],
        "edit_prompt": ctx["edit_prompt"],
        "raw_diagram_url": result["raw_diagram_url"],
        "diagram_url": result["diagram_url"],
        "diagram_error": result["diagram_error"],
        "beauty_applied": result["beauty_applied"],
        "beauty_enabled": False,
        "model_candidates": result["model_candidates"],
        "esp_push_ok": esp["esp_push_ok"],
        "esp_push_msg": esp["esp_push_msg"],
        "esp_pushed_image": esp["esp_target_image"] if esp["esp_push_ok"] else "",
        "esp_push_count": esp["esp_push_count"],
        "esp_push_total": esp["esp_push_total"],
    }


def build_response(ctx: Dict[str, Any], result: Dict[str, Any], esp: Dict[str, Any]) -> Dict[str, Any]:
    """返回给前端的 JSON 体（键集与改前逐字相同：无 time/filename）。"""
    session_data = ctx["session_data"]
    return {
        "ok": True,
        "msg": "已根据你的目标生成专业策略和理想结果图。",
        "session_id": ctx["session_id"],
        "stage": "generated",
        "preview_url": session_data.get("preview_url", ""),
        "vision_model": session_data.get("vision_model", ""),
        "image_model": result["image_model"],
        "diagnosis_report": ctx["diagnosis_report"],
        "user_intent": ctx["user_intent"],
        "generation_settings": ctx["generation_settings"],
        "strategy_plan": ctx["strategy_plan"],
        "ai_result": ctx["final_guidance"],
        "raw_diagram_url": result["raw_diagram_url"],
        "diagram_url": result["diagram_url"],
        "diagram_error": result["diagram_error"],
        "beauty_applied": result["beauty_applied"],
        "beauty_enabled": False,
        "model_candidates": result["model_candidates"],
        "esp_push_ok": esp["esp_push_ok"],
        "esp_push_msg": esp["esp_push_msg"],
        "esp_pushed_image": esp["esp_target_image"] if esp["esp_push_ok"] else "",
        "esp_push_count": esp["esp_push_count"],
        "esp_push_total": esp["esp_push_total"],
    }
