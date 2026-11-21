#!/usr/bin/env python3
"""契约快照/性能取数驱动（legacy 与新树同一脚本）。

用法（cwd 必须是 P.A）：
    PYTHONIOENCODING=utf-8 .venv/bin/python tests/run_snapshot.py --target legacy --out tests/baseline
    PYTHONIOENCODING=utf-8 .venv/bin/python tests/run_snapshot.py --target new    --out tests/phase1_snap

输出：snapshots.json / route_table.txt / values.json / push_logs.json / perf.json / bodies/
纪律：requests.* tripwire 先装（付费接口零调用）；磁盘写入全走沙盒；不执行 __main__。
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import snapshot_lib as sl  # noqa: E402

# 罐头桩值：ai_result 键与 normalize_ai_result 输出（11 键）同形，
# compare 结果与 run_model_compare 成功返回同形。
AI_RESULT = {
    "scene_summary": "stub-scene",
    "main_issue": "stub-issue",
    "subject_gender": "stub-gender",
    "subject_position_analysis": "stub-position",
    "camera_angle_analysis": "stub-angle",
    "shot_size_analysis": "stub-shot-size",
    "composition_analysis": "stub-composition",
    "light_source_inference": "stub-light",
    "recommended_shooting_position": "stub-rec",
    "suggested_adjustment": "stub-adjust",
    "ideal_image_prompt": "stub-ideal",
}

CANDS = ["wan2.7-image", "doubao-seedream-5.0", "qwen-image-2.0"]
COMPARE_RESULT = {
    "diagram_url": "http://stub.local/gen_main.png",
    "raw_diagram_url": "http://stub.local/raw_main.png",
    "image_model": CANDS[0],
    "diagram_error": "",
    "beauty_applied": False,
    "model_candidates": [
        {
            "model": m,
            "raw_diagram_url": f"http://stub.local/raw_{i}.png",
            "diagram_url": f"http://stub.local/gen_{i}.png",
            "beauty_applied": False,
            "error": "",
            "quality_score": 0.9 - 0.1 * i,
            "quality_reason": "stub-quality",
        }
        for i, m in enumerate(CANDS)
    ],
}

USER_INTENT = {
    "optimization_goal": "auto",
    "shot_type": "half_body",
    "mood_style": "natural",
    "output_ratio": "screen_4_3",
    "edit_strength": "standard",
}

FIXTURE_BYTES = sl.FIXTURE_SRC.read_bytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["legacy", "new"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bodies").mkdir(exist_ok=True)

    sl.install_tripwire()                      # 防护先装
    flask_app, mods = sl.load_target(args.target)
    sandbox = sl.make_sandbox(out / "sandbox")
    sl.apply_redirects(mods, sandbox, flask_app)
    sl.apply_stubs(mods, sl.make_stubs(AI_RESULT, COMPARE_RESULT))

    client = flask_app.test_client()
    snaps, values, push_logs = {}, {}, {}

    def snap(name, resp, body_save=None):
        snaps[name] = sl.capture(resp)
        if body_save:
            (out / "bodies" / body_save).write_bytes(resp.get_data())

    # ---- 页面 ----
    snap("GET /", client.get("/"), "index.html")
    snap("GET /mobile", client.get("/mobile"), "mobile.html")
    snap("GET /display", client.get("/display"), "display.html")
    snap("GET /qrcode", client.get("/qrcode"), "qrcode.html")
    # 静态目录接线探针：Flask static_folder 为构造期实参（真实路径，不经沙盒）
    snap("GET /static/latest_result.json", client.get("/static/latest_result.json"))

    # ---- esp_config（写路径全在沙盒）----
    snap("GET /api/esp_config", client.get("/api/esp_config"))
    snap("POST /api/esp_config esp_enabled",
         client.post("/api/esp_config", json={"esp_enabled": True}))
    snap("POST /api/esp_config bad ip",
         client.post("/api/esp_config", json={"esp_ip": "999.1.1.1"}))
    snap("POST /api/esp_config set ip",
         client.post("/api/esp_config", json={"esp_ip": "192.168.4.10"}))
    r = client.get("/api/esp_config")
    snap("GET /api/esp_config after", r)
    values["esp_config_after"] = r.get_json()   # 仅 esp_* 字段，无密钥

    # ---- diagnose ----
    r = client.post("/api/diagnose", data={
        "photo": (io.BytesIO(FIXTURE_BYTES), "photo.jpg"),
        "mode": "standard",
    }, content_type="multipart/form-data")
    snap("POST /api/diagnose", r)
    session_id = (r.get_json() or {}).get("session_id", "")
    values["diagnose_session_id"] = session_id
    snap("POST /api/diagnose no photo",
         client.post("/api/diagnose", data={}, content_type="multipart/form-data"))

    # ---- generate（compare 分支 / 单模型分支 / 未确认）----
    gen_body = {"generation_confirmed": True, "session_id": session_id,
                "user_intent": dict(USER_INTENT)}
    snap("POST /api/generate compare",
         client.post("/api/generate", json=dict(gen_body, debug_compare=True)))
    snap("POST /api/generate single",
         client.post("/api/generate", json=dict(gen_body, debug_compare=False)))
    snap("POST /api/generate unconfirmed",
         client.post("/api/generate", json={"session_id": session_id}))

    # ---- push（含 D5 双循环分歧探针）----
    sl.PUSH_LOG.clear()
    sl._PUSH_FAIL_ON[0] = None
    r = client.post("/api/push_latest_to_esp", json={"image_ref": "", "include_all": True})
    snap("POST /api/push_latest_to_esp fallback", r)
    push_logs["push_fallback"] = list(sl.PUSH_LOG)

    # 非当前结果图应被拒（:3073 防护分支）
    snap("POST /api/push_latest_to_esp reject foreign",
         client.post("/api/push_latest_to_esp",
                     json={"image_ref": "http://evil.example/x.png", "include_all": True}))

    # D5 分歧对打：先 compare 生成刷新 latest（3 候选），两侧用同一失败序（第 2 次起失败）
    sl.PUSH_LOG.clear()
    sl._PUSH_FAIL_ON[0] = 2
    r = client.post("/api/generate", json=dict(gen_body, debug_compare=True))
    snap("POST /api/generate fail-probe", r)
    push_logs["generate_fail_probe_break"] = list(sl.PUSH_LOG)   # 预期 2 次尝试（首败即 break）
    sl._PUSH_FAIL_ON[0] = None

    sl.PUSH_LOG.clear()
    sl._PUSH_FAIL_ON[0] = 2
    r = client.post("/api/push_latest_to_esp",
                    json={"image_ref": "http://stub.local/gen_main.png", "include_all": True})
    snap("POST /api/push_latest_to_esp fail-probe", r)
    push_logs["push_fail_probe_no_break"] = list(sl.PUSH_LOG)    # 预期 4 次尝试（无 break）
    sl._PUSH_FAIL_ON[0] = None

    # ---- latest / uploads / upload ----
    snap("GET /api/latest_result", client.get("/api/latest_result"))
    snap("GET /uploads/fixture.jpg", client.get("/uploads/fixture.jpg"))
    snap("POST /upload", client.post("/upload", data={
        "photo": (io.BytesIO(FIXTURE_BYTES), "photo.jpg"),
        "mode": "standard",
    }, content_type="multipart/form-data"))

    # ---- 性能 ----
    t0 = time.perf_counter()
    for _ in range(20):
        assert client.get("/api/esp_config").status_code == 200
    perf = {"esp_config_x20_ms": round((time.perf_counter() - t0) * 1000, 1)}

    with sl.counting() as c:
        assert client.get("/api/esp_config").status_code == 200
    perf["normal_request"] = {"open_calls": c.open_calls,
                              "api_key_opens": c.api_key_opens,
                              "subprocess_calls": c.subprocess_calls}

    # 最坏情况：空 api_keys.json（全部缺键）→ powershell 回退
    Path(sandbox["API_KEYS_FILE"]).write_text("{}", encoding="utf-8")
    with sl.counting() as c:
        assert client.get("/api/esp_config").status_code == 200
    perf["worst_case_request"] = {"open_calls": c.open_calls,
                                  "api_key_opens": c.api_key_opens,
                                  "subprocess_calls": c.subprocess_calls}

    # ---- 落盘 ----
    (out / "route_table.txt").write_text(sl.route_table(flask_app), encoding="utf-8")
    (out / "snapshots.json").write_text(
        json.dumps(snaps, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "values.json").write_text(
        json.dumps(values, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "push_logs.json").write_text(
        json.dumps(push_logs, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "perf.json").write_text(
        json.dumps(perf, ensure_ascii=False, indent=1), encoding="utf-8")
    print("OK", args.target, "->", out, "| scenarios:", len(snaps))


if __name__ == "__main__":
    main()
