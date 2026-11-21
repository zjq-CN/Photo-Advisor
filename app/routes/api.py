"""API 路由。"""

# 阶段① 原样搬迁；阶段③ 把 api_generate 拆到 app/generation.py，推送循环收进 app/esp.push_queue。
from typing import Any, Dict
from datetime import datetime
from geometry import extract_geometry  # 几何结构层：mediapipe 缺失时自动降级为 None
import json
from flask import request, jsonify
import os
import requests
import time
import uuid
from app.config import LATEST_RESULT_JSON, get_runtime_esp_config, is_valid_ipv4, parse_bool, update_api_keys_file
from app.contracts import VISION_MODE_PRESETS
from app.esp import build_esp_push_queue, push_queue
from app.generation import (
    apply_to_session, build_latest_payload, build_response,
    prepare_generate_request, push_generated, run_generation_models,
)
from app.server import app
from app.storage import build_upload_filename, load_session_data, save_latest_result, save_session_data
from app.vision import call_vision_auto



@app.route("/api/esp_config", methods=["GET", "POST"])
def api_esp_config():
    try:
        if request.method == "GET":
            cfg = get_runtime_esp_config()
            return jsonify({
                "ok": True,
                "esp_enabled": cfg["enabled"],
                "esp_ip": cfg["ip"],
                "esp_port": cfg["port"],
                "esp_endpoint": cfg["endpoint"],
                "esp_size": cfg["size"],
            })

        payload = request.get_json(silent=True) or {}
        updates: Dict[str, str] = {}
        if "esp_enabled" in payload:
            raw_enabled = payload.get("esp_enabled")
            enabled = raw_enabled if isinstance(raw_enabled, bool) else parse_bool(str(raw_enabled), True)
            updates["ESP_SCREEN_ENABLED"] = "1" if enabled else "0"
        esp_ip = str(payload.get("esp_ip", "")).strip()
        if esp_ip:
            if not is_valid_ipv4(esp_ip):
                return jsonify({"ok": False, "msg": "ESP IP 格式不正确，请填写 IPv4 地址"}), 400
            updates["ESP_SCREEN_IP"] = esp_ip
        if not updates:
            return jsonify({"ok": False, "msg": "没有可保存的 ESP 配置"}), 400

        update_api_keys_file(updates)
        cfg = get_runtime_esp_config()
        msg_parts = []
        if "ESP_SCREEN_IP" in updates:
            msg_parts.append(f"屏幕 IP：{updates['ESP_SCREEN_IP']}")
        if "ESP_SCREEN_ENABLED" in updates:
            msg_parts.append("小屏幕已启用" if updates["ESP_SCREEN_ENABLED"] == "1" else "小屏幕已关闭")
        saved_msg = "已保存。" + (" " + "；".join(msg_parts) if msg_parts else "")
        return jsonify({
            "ok": True,
            "msg": saved_msg,
            "esp_enabled": cfg["enabled"],
            "esp_ip": cfg["ip"],
            "esp_port": cfg["port"],
            "esp_endpoint": cfg["endpoint"],
            "esp_size": cfg["size"],
        })
    except Exception as e:
        return jsonify({"ok": False, "msg": f"保存 ESP 配置失败：{e}"}), 500


@app.route("/api/diagnose", methods=["POST"])
def api_diagnose():
    try:
        if "photo" not in request.files:
            return jsonify({"ok": False, "msg": "没有收到图片字段 photo"}), 400

        file = request.files["photo"]
        if file.filename == "":
            return jsonify({"ok": False, "msg": "未选择图片"}), 400

        new_filename = build_upload_filename(file.filename)

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], new_filename)
        file.save(save_path)

        print(f"[已接收] {save_path}")
        print("[开始基础摄影诊断]")

        mode = (request.form.get("mode") or "standard").strip()
        if mode not in VISION_MODE_PRESETS:
            mode = "standard"
        diagnosis_started = time.time()

        geo = extract_geometry(save_path)
        geometry_block = geo["prompt_block"] if geo else ""
        ai_result, used_vision_model = call_vision_auto(save_path, geometry_block, mode=mode)
        diagnosis_elapsed = round(time.time() - diagnosis_started, 1)
        print(f"[诊断完成] mode={mode} model={used_vision_model} 耗时={diagnosis_elapsed}s")
        preview_url = f"/uploads/{new_filename}"
        # 诊断阶段只保存原图，不推送到屏幕；屏幕仅展示 AI 生成结果。
        esp_push_ok, esp_push_msg = False, "等待 AI 生成后再推送到 ESP"
        print(f"[ESP] {esp_push_msg}")

        session_id = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:6]
        session_payload = {
            "session_id": session_id,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": new_filename,
            "preview_url": preview_url,
            "image_path": save_path,
            "vision_model": used_vision_model,
            "vision_mode": mode,
            "diagnosis_elapsed": diagnosis_elapsed,
            "detector_data": geo,
            "stage": "diagnosed",
            "diagnosis_report": ai_result,
            "user_intent": None,
            "strategy_plan": None,
            "edit_prompt": "",
            "raw_diagram_url": "",
            "diagram_url": "",
            "diagram_error": "",
            "image_model": "",
            "beauty_applied": False,
            "model_candidates": [],
            "esp_push_ok": esp_push_ok,
            "esp_push_msg": esp_push_msg,
            "esp_pushed_image": "",
            "esp_push_count": 0,
            "esp_push_total": 0,
        }
        save_session_data(session_id, session_payload)

        latest_payload = {
            "ok": True,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "stage": "diagnosed",
            "filename": new_filename,
            "preview_url": preview_url,
            "vision_model": used_vision_model,
            "vision_mode": mode,
            "diagnosis_elapsed": diagnosis_elapsed,
            "detector_data": geo,
            "ai_result": ai_result,
            "diagnosis_report": ai_result,
            "strategy_plan": None,
            "user_intent": None,
            "edit_prompt": "",
            "raw_diagram_url": "",
            "diagram_url": "",
            "diagram_error": "",
            "image_model": "",
            "beauty_applied": False,
            "model_candidates": [],
            "esp_push_ok": esp_push_ok,
            "esp_push_msg": esp_push_msg,
            "esp_pushed_image": "",
            "esp_push_count": 0,
            "esp_push_total": 0,
        }
        save_latest_result(latest_payload)

        return jsonify({
            "ok": True,
            "msg": "拍照上传成功，已完成基础摄影诊断。",
            "session_id": session_id,
            "stage": "diagnosed",
            "filename": new_filename,
            "preview_url": preview_url,
            "vision_model": used_vision_model,
            "vision_mode": mode,
            "diagnosis_elapsed": diagnosis_elapsed,
            "ai_result": ai_result,
            "diagnosis_report": ai_result,
            "esp_push_ok": esp_push_ok,
            "esp_push_msg": esp_push_msg,
            "esp_pushed_image": "",
            "esp_push_count": 0,
            "esp_push_total": 0,
        })

    except ValueError as e:
        return jsonify({"ok": False, "msg": str(e)}), 400

    except requests.HTTPError as e:
        err_text = ""
        try:
            err_text = e.response.text
        except Exception:
            err_text = str(e)

        latest_payload = {
            "ok": False,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": f"接口 HTTP 错误：{err_text}"
        }
        save_latest_result(latest_payload)
        return jsonify({"ok": False, "msg": "模型接口调用失败，请查看电脑端日志。"}), 500

    except Exception as e:
        latest_payload = {
            "ok": False,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": f"处理失败：{str(e)}"
        }
        save_latest_result(latest_payload)
        return jsonify({"ok": False, "msg": f"处理失败：{str(e)}"}), 500


@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        payload = request.get_json(silent=True) or {}
        err, ctx = prepare_generate_request(payload)
        if err is not None:
            return err

        print("[开始图像编辑生成理想拍摄结果图]")
        result = run_generation_models(ctx)
        esp = push_generated(ctx, result)

        apply_to_session(ctx, result, esp)
        save_session_data(ctx["session_id"], ctx["session_data"])

        save_latest_result(build_latest_payload(ctx, result, esp))
        return jsonify(build_response(ctx, result, esp))

    except FileNotFoundError as e:
        return jsonify({"ok": False, "msg": str(e)}), 404

    except requests.HTTPError as e:
        err_text = ""
        try:
            err_text = e.response.text
        except Exception:
            err_text = str(e)
        return jsonify({"ok": False, "msg": f"模型接口调用失败：{err_text}"}), 500

    except Exception as e:
        return jsonify({"ok": False, "msg": f"处理失败：{str(e)}"}), 500


@app.route("/api/push_latest_to_esp", methods=["POST"])
def api_push_latest_to_esp():
    try:
        payload = request.get_json(silent=True) or {}
        image_ref = str(payload.get("image_ref", "")).strip()
        include_all = parse_bool(str(payload.get("include_all", "0")), False)

        latest_data: Dict[str, Any] = {}
        if os.path.exists(LATEST_RESULT_JSON):
            with open(LATEST_RESULT_JSON, "r", encoding="utf-8") as f:
                latest_data = json.load(f)

        allowed_refs = {
            str(latest_data.get("diagram_url", "")).strip(),
            str(latest_data.get("raw_diagram_url", "")).strip(),
        }
        latest_candidates = latest_data.get("model_candidates", [])
        if isinstance(latest_candidates, list):
            for item in latest_candidates:
                if isinstance(item, dict):
                    allowed_refs.add(str(item.get("diagram_url", "")).strip())
                    allowed_refs.add(str(item.get("raw_diagram_url", "")).strip())
        allowed_refs.discard("")

        if image_ref and image_ref not in allowed_refs:
            return jsonify({"ok": False, "msg": "拒绝推送非当前生成结果的图片。"}), 400

        if not image_ref:
            image_ref = str(latest_data.get("diagram_url", "")).strip()

        if not image_ref:
            return jsonify({"ok": False, "msg": "暂无可推送 AI 生成图，请先完成生成。"}), 400

        candidates = latest_data.get("model_candidates", []) if include_all else []
        if not isinstance(candidates, list):
            candidates = []
        queue = build_esp_push_queue(image_ref, candidates, include_all=include_all)
        if not queue:
            return jsonify({"ok": False, "msg": "暂无可推送 AI 生成图，请先完成生成。"}), 400

        ok, msg, success_count, _total = push_queue(
            queue, stop_on_first_failure=False, add_skip_note=False)

        if latest_data:
            selected_candidate = None
            if isinstance(latest_candidates, list):
                selected_candidate = next(
                    (
                        item for item in latest_candidates
                        if isinstance(item, dict)
                        and str(item.get("diagram_url", "")).strip() == image_ref
                    ),
                    None,
                )
            if ok:
                latest_data["diagram_url"] = image_ref
                if isinstance(selected_candidate, dict):
                    latest_data["raw_diagram_url"] = selected_candidate.get("raw_diagram_url", image_ref)
                    latest_data["image_model"] = selected_candidate.get("model", latest_data.get("image_model", ""))
            latest_data["esp_push_ok"] = ok
            latest_data["esp_push_msg"] = msg
            latest_data["esp_pushed_image"] = image_ref if ok else ""
            latest_data["esp_push_count"] = success_count
            latest_data["esp_push_total"] = len(queue)
            save_latest_result(latest_data)

            session_id = str(latest_data.get("session_id", "")).strip()
            if ok and session_id:
                try:
                    session_data = load_session_data(session_id)
                    session_data["diagram_url"] = latest_data.get("diagram_url", image_ref)
                    session_data["raw_diagram_url"] = latest_data.get("raw_diagram_url", image_ref)
                    session_data["image_model"] = latest_data.get("image_model", "")
                    session_data["esp_pushed_image"] = image_ref
                    save_session_data(session_id, session_data)
                except Exception as e:
                    print(f"[ESP] 更新会话主图失败：{e}")

        return jsonify({
            "ok": ok,
            "msg": msg,
            "image_ref": image_ref,
            "push_count": success_count,
            "push_total": len(queue),
        }), 200 if ok else 500
    except Exception as e:
        return jsonify({"ok": False, "msg": f"ESP 推送失败：{e}"}), 500


@app.route("/api/latest_result")
def api_latest_result():
    if not os.path.exists(LATEST_RESULT_JSON):
        return jsonify({
            "ok": False,
            "msg": "暂无数据，请先用手机拍照上传。"
        })

    with open(LATEST_RESULT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "raw_diagram_url" not in data:
        data["raw_diagram_url"] = data.get("diagram_url", "")
    if "beauty_applied" not in data:
        data["beauty_applied"] = False
    if "beauty_enabled" not in data:
        data["beauty_enabled"] = False
    if "model_candidates" not in data:
        data["model_candidates"] = []
    if "esp_push_ok" not in data:
        data["esp_push_ok"] = False
    if "esp_push_msg" not in data:
        data["esp_push_msg"] = ""
    if "esp_pushed_image" not in data:
        data["esp_pushed_image"] = ""
    if "esp_push_count" not in data:
        data["esp_push_count"] = 0
    if "esp_push_total" not in data:
        data["esp_push_total"] = 0
    return jsonify(data)
