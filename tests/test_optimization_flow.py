import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
# geometry.py 位于项目根目录，主程序顶层会 `from geometry import extract_geometry`；
# importlib 加载不会自动把项目根加入 sys.path，这里手动补上。
sys.path.insert(0, str(PROJECT_DIR))
PRIMARY_PATH = PROJECT_DIR / "testV1.0_backup_prompt.py"

# 避免测试导入阶段为未配置项反复查询 Windows 用户环境；这些值只存在于
# 当前测试进程，所有 HTTP 调用仍会被 mock，绝不会发起付费请求。
for _name in (
    "DASHSCOPE_API_KEY",
    "DOUBAO_API_KEY",
    "ZHIPU_API_KEY",
    "JIMENG_ACCESS_KEY_ID",
    "JIMENG_SECRET_ACCESS_KEY",
    "JIMENG_IMAGE_URL",
    "HUNYUAN_API_KEY",
    "HUNYUAN_SECRET_ID",
    "HUNYUAN_SECRET_KEY",
):
    os.environ.setdefault(_name, "unit-test-only")
os.environ.setdefault("ESP_SCREEN_ENABLED", "0")

_spec = importlib.util.spec_from_file_location("photo_teaching_app", PRIMARY_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"无法加载主程序：{PRIMARY_PATH}")
app_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app_module)


DIAGNOSIS = {
    "scene_summary": "一名人物站在室内窗边。",
    "subject_position_analysis": "人物略偏右，主体占比适中。",
    "camera_angle_analysis": "机位基本平视。",
    "shot_size_analysis": "当前为半身人像。",
    "composition_analysis": "头顶留白稍多，画面左侧有少量干扰。",
    "light_source_inference": "面部曝光略暗，窗光方向可判断。",
    "suggested_adjustment": "轻微裁切并恢复面部暗部。",
    "ideal_image_prompt": "一张自然真实的半身照片。",
}


def _build_prompt(intent):
    strategy = app_module.build_strategy_plan(DIAGNOSIS, intent)
    target = app_module.build_targeted_ideal_prompt(DIAGNOSIS, strategy, intent)
    return app_module.build_edit_prompt(DIAGNOSIS, strategy, intent, target)


class OptimizationSchemaTests(unittest.TestCase):
    def test_schema_defaults_are_valid_and_unique(self):
        self.assertEqual(
            [group["key"] for group in app_module.OPTIMIZATION_SCHEMA],
            ["optimization_goal", "shot_type", "mood_style", "output_ratio", "edit_strength"],
        )
        for group in app_module.OPTIMIZATION_SCHEMA:
            values = [option["value"] for option in group["options"]]
            self.assertEqual(len(values), len(set(values)))
            self.assertIn(app_module.DEFAULT_USER_INTENT[group["key"]], values)

    def test_model_order_is_the_recommended_three(self):
        self.assertEqual(
            app_module.IMAGE_EDIT_MODELS,
            [
                "wan2.7-image",
                "doubao-seedream-5.0",
                "qwen-image-2.0",
            ],
        )

    def test_invalid_values_fall_back_and_legacy_values_migrate(self):
        normalized = app_module.normalize_user_intent({"optimization_goal": "not-real"})
        self.assertEqual(normalized, app_module.DEFAULT_USER_INTENT)

        legacy = app_module.normalize_user_intent({
            "visual_goal": "人物更突出",
            "emotion_style": "温暖",
            "usage_type": "职业头像",
        })
        self.assertEqual(legacy["optimization_goal"], "subject")
        self.assertEqual(legacy["mood_style"], "warm")
        self.assertEqual(legacy["shot_type"], "headshot")

    def test_every_non_default_option_changes_the_final_prompt(self):
        default_intent = dict(app_module.DEFAULT_USER_INTENT)
        default_prompt = _build_prompt(default_intent)

        for group in app_module.OPTIMIZATION_SCHEMA:
            key = group["key"]
            for option in group["options"]:
                if option["value"] == default_intent[key]:
                    continue
                with self.subTest(group=key, option=option["value"]):
                    intent = dict(default_intent)
                    intent[key] = option["value"]
                    prompt = _build_prompt(intent)
                    self.assertNotEqual(prompt, default_prompt)
                    self.assertIn(option["label"], prompt)

    def test_auto_goal_uses_diagnosis(self):
        intent = dict(app_module.DEFAULT_USER_INTENT)
        lighting_diagnosis = dict(DIAGNOSIS)
        lighting_diagnosis.update({
            "subject_position_analysis": "人物位置稳定。",
            "composition_analysis": "构图基本合理。",
            "light_source_inference": "画面严重过暗，曝光不足，阴影死黑且白平衡偏色。",
            "suggested_adjustment": "优先恢复曝光、暗部和真实肤色。",
        })
        strategy = app_module.build_strategy_plan(lighting_diagnosis, intent)
        self.assertEqual(strategy["requested_goal"], "auto")
        self.assertEqual(strategy["effective_goal"], "lighting")
        target = app_module.build_targeted_ideal_prompt(lighting_diagnosis, strategy, intent)
        prompt = app_module.build_edit_prompt(lighting_diagnosis, strategy, intent, target)
        self.assertIn("光线与肤色", prompt)

    def test_generation_sizes_follow_ratio(self):
        expected = {
            "source": (None, "2K", None),
            "screen_4_3": ("2048*1536", "2560x1920", 4 / 3),
            "portrait_3_4": ("1536*2048", "1920x2560", 3 / 4),
            "square_1_1": ("2048*2048", "2048x2048", 1.0),
            "landscape_16_9": ("2048*1152", "2560x1440", 16 / 9),
        }
        for ratio, values in expected.items():
            with self.subTest(ratio=ratio):
                intent = dict(app_module.DEFAULT_USER_INTENT, output_ratio=ratio)
                settings = app_module.build_generation_settings(intent)
                self.assertEqual(
                    (settings["dashscope_size"], settings["doubao_size"], settings["expected_ratio"]),
                    values,
                )

    def test_lighting_goal_and_mood_do_not_overwrite_each_other(self):
        intent = dict(
            app_module.DEFAULT_USER_INTENT,
            optimization_goal="lighting",
            mood_style="warm",
        )
        strategy = app_module.build_strategy_plan(DIAGNOSIS, intent)
        self.assertIn("白平衡", strategy["color_strategy"])
        self.assertIn("轻微增暖", strategy["color_strategy"])

    def test_session_id_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            app_module.validate_session_id("../../api_keys")

    def test_local_ip_parser_supports_english_windows_output(self):
        output = """Windows IP Configuration

Ethernet adapter vEthernet (Default Switch):
   IPv4 Address. . . . . . . . . . . : 172.20.0.1

Wireless LAN adapter WLAN:
   IPv4 Address. . . . . . . . . . . : 192.168.43.26
   Default Gateway . . . . . . . . . : 192.168.43.1
"""
        completed = Mock(stdout=output)
        with patch.object(app_module.subprocess, "run", return_value=completed):
            self.assertEqual(app_module.get_local_ip(), "192.168.43.26")


class RenderingAndPayloadTests(unittest.TestCase):
    def test_mobile_page_is_rendered_from_backend_schema(self):
        with app_module.app.test_client() as client:
            response = client.get("/mobile")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        match = re.search(r'<script type="application/json" id="intentSchemaData">(.*?)</script>', html, re.DOTALL)
        self.assertIsNotNone(match)
        rendered_schema = json.loads(match.group(1))
        rendered_text = json.dumps(rendered_schema, ensure_ascii=False)
        for label in ("优先改善", "目标景别", "输出画幅", "优化强度", "4:3 设备屏幕"):
            self.assertIn(label, rendered_text)
        self.assertNotIn("使用场景", rendered_text)
        self.assertNotIn("情绪风格", rendered_text)

    def test_display_page_preserves_candidate_aspect_ratio(self):
        with app_module.app.test_client() as client:
            response = client.get("/display")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertNotIn("aspect-ratio: 3 / 4", html)
        self.assertIn("object-fit: contain", html)
        self.assertIn("推送为屏幕主图", html)
        self.assertIn("本次实际优化", html)

    def test_screen_conversion_preserves_portrait_orientation(self):
        portrait = np.zeros((200, 100, 3), dtype=np.uint8)
        portrait[:, :] = (20, 180, 40)
        ok, encoded = cv2.imencode(".png", portrait)
        self.assertTrue(ok)

        jpeg, rotated = app_module.build_screen_jpeg(encoded.tobytes(), 320, 240, 90)
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertFalse(rotated)
        self.assertEqual(decoded.shape[:2], (240, 320))
        self.assertGreater(float(decoded[:, 150:170].mean()), 40.0)
        self.assertLess(float(decoded[:, :50].mean()), 8.0)
        self.assertLess(float(decoded[:, -50:].mean()), 8.0)

    def test_dashscope_payload_contains_requested_size(self):
        fake_response = Mock(status_code=200, text='{"ok":true}')
        fake_response.json.return_value = {
            "output": {"choices": [{"message": {"content": [{"image": "https://example.test/qwen.jpg"}]}}]}
        }
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            image_path = handle.name
            handle.write(b"test-image")
        try:
            with patch.object(app_module.requests, "post", return_value=fake_response) as post:
                result = app_module.edit_with_qwen_image_20(
                    image_path,
                    "test prompt",
                    {"dashscope_size": "2048*1536"},
                )
            self.assertEqual(result, "https://example.test/qwen.jpg")
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["parameters"]["size"], "2048*1536")
            self.assertFalse(payload["parameters"]["prompt_extend"])
        finally:
            Path(image_path).unlink(missing_ok=True)

    def test_seedream_payload_uses_reference_array_and_requested_size(self):
        fake_response = Mock(status_code=200, text='{"ok":true}')
        fake_response.json.return_value = {"data": [{"url": "https://example.test/seedream.jpg"}]}
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            image_path = handle.name
            handle.write(b"test-image")
        try:
            with patch.object(app_module, "build_doubao_model_candidates", return_value=["seedream-test"]), patch.object(
                app_module.requests, "post", return_value=fake_response
            ) as post:
                result = app_module.edit_with_doubao_seedream(
                    image_path,
                    "test prompt",
                    {"doubao_size": "1536x2048"},
                )
            self.assertEqual(result, "https://example.test/seedream.jpg")
            payload = post.call_args.kwargs["json"]
            self.assertIsInstance(payload["image"], list)
            self.assertEqual(payload["size"], "1536x2048")
            self.assertEqual(payload["sequential_image_generation"], "disabled")
        finally:
            Path(image_path).unlink(missing_ok=True)

    def test_push_endpoint_rejects_unknown_image_and_accepts_current_candidate(self):
        latest = {
            "ok": True,
            "diagram_url": "/uploads/a.jpg",
            "raw_diagram_url": "/uploads/a.jpg",
            "model_candidates": [
                {"model": "qwen-image-2.0", "diagram_url": "/uploads/b.jpg", "raw_diagram_url": "/uploads/b.jpg"}
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            latest_path = Path(temp_dir) / "latest.json"
            latest_path.write_text(json.dumps(latest, ensure_ascii=False), encoding="utf-8")
            with patch.object(app_module, "LATEST_RESULT_JSON", str(latest_path)), patch.object(
                app_module, "push_image_to_esp", return_value=(True, "mock pushed")
            ) as push:
                with app_module.app.test_client() as client:
                    rejected = client.post(
                        "/api/push_latest_to_esp",
                        json={"image_ref": "https://evil.example/unknown.jpg"},
                    )
                    accepted = client.post(
                        "/api/push_latest_to_esp",
                        json={"image_ref": "/uploads/b.jpg"},
                    )
            self.assertEqual(rejected.status_code, 400)
            self.assertEqual(accepted.status_code, 200)
            push.assert_called_once_with("/uploads/b.jpg", append=False)
            updated = json.loads(latest_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["diagram_url"], "/uploads/b.jpg")
            self.assertEqual(updated["image_model"], "qwen-image-2.0")

    def test_generate_route_wires_options_prompt_size_and_candidate_result(self):
        session_id = "20260816123456_abc123"
        selected_intent = {
            "optimization_goal": "lighting",
            "shot_type": "headshot",
            "mood_style": "warm",
            "output_ratio": "portrait_3_4",
            "edit_strength": "strong",
        }
        compare_result = {
            "diagram_url": "/uploads/generated.jpg",
            "raw_diagram_url": "/uploads/generated.jpg",
            "image_model": "qwen-image-2.0",
            "diagram_error": "",
            "beauty_applied": False,
            "model_candidates": [{
                "model": "qwen-image-2.0",
                "diagram_url": "/uploads/generated.jpg",
                "raw_diagram_url": "/uploads/generated.jpg",
                "error": "",
            }],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_dir = root / "sessions"
            session_dir.mkdir()
            source_image = root / "source.jpg"
            source_image.write_bytes(b"local-test-image")
            session_payload = {
                "session_id": session_id,
                "image_path": str(source_image),
                "preview_url": "/uploads/source.jpg",
                "filename": "source.jpg",
                "vision_model": "vision-test",
                "diagnosis_report": DIAGNOSIS,
            }
            (session_dir / f"{session_id}.json").write_text(
                json.dumps(session_payload, ensure_ascii=False),
                encoding="utf-8",
            )
            latest_path = root / "latest.json"

            with patch.object(app_module, "SESSION_FOLDER", str(session_dir)), patch.object(
                app_module, "LATEST_RESULT_JSON", str(latest_path)
            ), patch.object(
                app_module, "run_model_compare", return_value=compare_result
            ) as compare, patch.object(
                app_module, "push_image_to_esp", return_value=(True, "mock pushed")
            ):
                with app_module.app.test_client() as client:
                    response = client.post(
                        "/api/generate",
                        json={
                            "session_id": session_id,
                            "user_intent": selected_intent,
                            "generation_confirmed": True,
                            "debug_compare": True,
                        },
                    )

            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertTrue(data["ok"])
            self.assertEqual(data["user_intent"], selected_intent)
            self.assertEqual(data["generation_settings"]["dashscope_size"], "1536*2048")
            self.assertEqual(data["image_model"], "qwen-image-2.0")
            called_prompt = compare.call_args.args[1]
            for label in ("光线与肤色", "头像特写", "温暖柔和", "3:4 竖版人像", "明显"):
                self.assertIn(label, called_prompt)


class GeometryInjectionTests(unittest.TestCase):
    """几何结构层集成：提示词追加与 /api/diagnose 载荷。"""

    def test_analysis_prompt_without_block_matches_original(self):
        prompt = app_module.build_analysis_prompt()
        for key in ("你是专业的人像摄影指导", "只输出一个 JSON 对象", "main_issue", "scene_summary", "ideal_image_prompt"):
            self.assertIn(key, prompt)
        self.assertNotIn("【检测器参考数据】", prompt)

    def test_analysis_prompt_appends_geometry_block_at_end(self):
        block = "【检测器参考数据】\n- 画面中检测到至少 2 张人脸。"
        prompt = app_module.build_analysis_prompt(block)
        self.assertTrue(prompt.endswith(block))
        self.assertIn("【检测器参考数据】", prompt)

    def test_dashscope_branch_payload_contains_geometry_block(self):
        fake_response = Mock(status_code=200, text='{"ok":true}')
        fake_response.json.return_value = {
            "choices": [{"message": {"content": '{"scene_summary": "测试"}'}}]
        }
        block = "【检测器参考数据】\n- 画面中检测到至少 1 张人脸。"
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(b"fake-image-bytes")
            image_path = handle.name
        try:
            with patch.object(app_module.requests, "post", return_value=fake_response) as post:
                app_module.call_vision_once("dashscope-test", image_path, block)
            payload = post.call_args.kwargs["json"]
            prompt_text = payload["messages"][0]["content"][0]["text"]
            self.assertTrue(prompt_text.endswith(block))
        finally:
            Path(image_path).unlink(missing_ok=True)

    def test_api_diagnose_injects_detector_block_and_persists_detector_data(self):
        fake_geo = {
            "detector": "mediapipe_face_mesh",
            "count_note": "画面中检测到至少 1 张人脸",
            "people": [
                {
                    "index": 1,
                    "x_norm": 0.5,
                    "y_norm": 0.4,
                    "face_width_ratio": 0.12,
                    "yaw_deg": 0.0,
                    "pitch_deg": 0.0,
                    "roll_deg": 0.0,
                    "position_x": "居中",
                    "position_y": "中上",
                    "face_size": "中等",
                    "head_pose": "正面",
                    "roll_desc": "端正",
                    "description": "人脸1：位于画面居中、中上，面朝镜头，面部占比中等，头部端正。",
                }
            ],
            "prompt_block": "【检测器参考数据】\n以下为计算机视觉检测器非人工标注的测量结果，仅供参考，最终以图像为准；人数为\"至少\"口径（人脸被遮挡可能漏检）。\n- 画面中检测到至少 1 张人脸。\n- 人脸1：位于画面居中、中上，正面朝向镜头，面部占比中等，头部端正。",
        }
        fake_ai_result = {"scene_summary": "一名人物站在室内窗边。"}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_dir = root / "sessions"
            session_dir.mkdir()
            upload_dir = root / "uploads"
            upload_dir.mkdir()
            latest_path = root / "latest.json"
            image_path = root / "photo.jpg"
            cv2.imwrite(str(image_path), np.zeros((60, 60, 3), dtype=np.uint8))
            image_bytes = image_path.read_bytes()
            with patch.dict(app_module.app.config, {"UPLOAD_FOLDER": str(upload_dir)}), patch.object(
                app_module, "SESSION_FOLDER", str(session_dir)
            ), patch.object(app_module, "LATEST_RESULT_JSON", str(latest_path)), patch.object(
                app_module, "extract_geometry", return_value=fake_geo
            ) as geo_call, patch.object(
                app_module, "call_vision_auto", return_value=(fake_ai_result, "vision-test")
            ) as vision_call:
                with app_module.app.test_client() as client:
                    response = client.post(
                        "/api/diagnose",
                        data={"photo": (io.BytesIO(image_bytes), "photo.jpg")},
                        content_type="multipart/form-data",
                    )
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertTrue(data["ok"])
            geo_call.assert_called_once()
            self.assertEqual(vision_call.call_args.args[1], fake_geo["prompt_block"])
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            self.assertEqual(latest["detector_data"], fake_geo)
            self.assertEqual(latest["detector_data"]["count_note"], "画面中检测到至少 1 张人脸")


class VisionModePresetTests(unittest.TestCase):
    """标准/快速双模式：preset 表、思考参数透传、/api/diagnose mode 字段。"""

    @staticmethod
    def _fake_response():
        fake_response = Mock(status_code=200, text='{"ok":true}')
        fake_response.json.return_value = {
            "choices": [{"message": {"content": '{"scene_summary": "测试"}'}}]
        }
        return fake_response

    def test_presets_shape_and_qwen_models_separated(self):
        presets = app_module.VISION_MODE_PRESETS
        self.assertEqual(set(presets), {"standard", "fast"})
        for preset in presets.values():
            self.assertEqual(len(preset["chain"]), 5)
            qwen_idx = [i for i, m in enumerate(preset["chain"]) if m.startswith("qwen")]
            for a, b in zip(qwen_idx, qwen_idx[1:]):
                self.assertGreater(b - a, 1, "两个百炼 qwen 模型必须被非百炼模型隔开")
        self.assertEqual(app_module.VISION_MODELS, presets["standard"]["chain"])

    def test_deepseek_thinking_param_passthrough(self):
        fake_response = self._fake_response()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(b"fake-image-bytes")
            image_path = handle.name
        try:
            with patch.object(app_module.requests, "post", return_value=fake_response) as post:
                app_module.call_vision_deepseek_once("deepseek-test", image_path, "", {"thinking": {"type": "disabled"}})
                self.assertEqual(post.call_args.kwargs["json"]["thinking"], {"type": "disabled"})
            with patch.object(app_module.requests, "post", return_value=fake_response) as post:
                app_module.call_vision_deepseek_once("deepseek-test", image_path, "")
                self.assertEqual(post.call_args.kwargs["json"]["thinking"], {"type": "enabled"})
        finally:
            os.unlink(image_path)

    def test_dashscope_enable_thinking_from_extra_params(self):
        fake_response = self._fake_response()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(b"fake-image-bytes")
            image_path = handle.name
        try:
            with patch.object(app_module.requests, "post", return_value=fake_response) as post:
                app_module.call_vision_once("dashscope-test", image_path, "", {"enable_thinking": True})
                self.assertTrue(post.call_args.kwargs["json"]["enable_thinking"])
            with patch.object(app_module.requests, "post", return_value=fake_response) as post:
                app_module.call_vision_once("dashscope-test", image_path, "")
                self.assertFalse(post.call_args.kwargs["json"]["enable_thinking"])
        finally:
            os.unlink(image_path)

    def test_zhipu_reasoning_effort_override(self):
        fake_response = self._fake_response()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(b"fake-image-bytes")
            image_path = handle.name
        try:
            with patch.object(app_module.requests, "post", return_value=fake_response) as post:
                app_module.call_vision_zhipu_once("glm-5.3-flash", image_path, "", {"reasoning_effort": "high"})
                self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "high")
            with patch.object(app_module.requests, "post", return_value=fake_response) as post:
                app_module.call_vision_zhipu_once("glm-5.3-flash", image_path, "")
                self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], app_module.ZHIPU_REASONING_EFFORT)
        finally:
            os.unlink(image_path)

    def test_api_diagnose_mode_passthrough_and_validation(self):
        fake_ai_result = {"scene_summary": "一名人物站在室内窗边。"}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_dir = root / "sessions"
            session_dir.mkdir()
            upload_dir = root / "uploads"
            upload_dir.mkdir()
            latest_path = root / "latest.json"
            image_path = root / "photo.jpg"
            cv2.imwrite(str(image_path), np.zeros((60, 60, 3), dtype=np.uint8))
            image_bytes = image_path.read_bytes()
            with patch.dict(app_module.app.config, {"UPLOAD_FOLDER": str(upload_dir)}), patch.object(
                app_module, "SESSION_FOLDER", str(session_dir)
            ), patch.object(app_module, "LATEST_RESULT_JSON", str(latest_path)), patch.object(
                app_module, "extract_geometry", return_value=None
            ), patch.object(
                app_module, "call_vision_auto", return_value=(fake_ai_result, "vision-test")
            ) as vision_call:
                with app_module.app.test_client() as client:
                    response = client.post(
                        "/api/diagnose",
                        data={"photo": (io.BytesIO(image_bytes), "photo.jpg"), "mode": "fast"},
                        content_type="multipart/form-data",
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(vision_call.call_args.kwargs["mode"], "fast")
                    self.assertEqual(response.get_json()["vision_mode"], "fast")

                    response = client.post(
                        "/api/diagnose",
                        data={"photo": (io.BytesIO(image_bytes), "photo.jpg"), "mode": "bogus"},
                        content_type="multipart/form-data",
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(vision_call.call_args.kwargs["mode"], "standard")
                    self.assertEqual(response.get_json()["vision_mode"], "standard")


if __name__ == "__main__":
    unittest.main()
