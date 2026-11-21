"""阶段① 契约快照/性能取数公共库。

legacy（testV1.0_backup_prompt.py）与新树（app/）共用同一套打桩与采集逻辑，
保证改前改后数字可比。纪律：
- 付费接口零调用：requests.* 打成 tripwire，任何真实 HTTP 立即抛错；
- 磁盘写入全部重定向到沙盒（uploads/static/sessions/latest_result/_config）；
- 绝不执行 __main__ 块（不 app.run、不开浏览器）。
"""
import builtins
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

PA_DIR = Path(__file__).resolve().parent.parent  # P.A/
FIXTURE_SRC = PA_DIR / "uploads" / "photo_20261114_193009_716673ba.jpg"

PUSH_LOG = []          # push_image_to_esp 打桩记录（观察 D5 双循环分歧的通道）
_PUSH_FAIL_ON = [None]  # [n]：第 n 次调用（1-based）返回失败，用于 break 分歧场景


# ---------------- 沙盒 ----------------
def make_sandbox(root: Path):
    root = Path(root).resolve()  # 必须绝对：Flask send_from_directory 对相对目录按 app.root_path 解析（假阳性源）
    (root / "_config").mkdir(parents=True, exist_ok=True)
    (root / "uploads").mkdir(parents=True, exist_ok=True)
    (root / "static" / "sessions").mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE_SRC, root / "uploads" / "fixture.jpg")
    shutil.copy(PA_DIR / "_config" / "api_keys.json", root / "_config" / "api_keys.json")
    return {
        "UPLOAD_FOLDER": str(root / "uploads"),
        "STATIC_FOLDER": str(root / "static"),
        "SESSION_FOLDER": str(root / "static" / "sessions"),
        "LATEST_RESULT_JSON": str(root / "static" / "latest_result.json"),
        "API_KEYS_FILE": str(root / "_config" / "api_keys.json"),
    }


# ---------------- 目标加载 ----------------
def load_target(target: str):
    """返回 (flask_app, 需要打桩/重定向的模块列表)。"""
    if str(PA_DIR) not in sys.path:
        sys.path.insert(0, str(PA_DIR))
    if target == "legacy":
        spec = importlib.util.spec_from_file_location(
            "pa_legacy", PA_DIR / "legacy" / "testV1.0_backup_prompt.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["pa_legacy"] = mod
        spec.loader.exec_module(mod)
        return mod.app, [mod]
    if target == "new":
        import app as app_pkg  # noqa: F401  （触发路由注册）
        from app.server import app as flask_app
        mods = [m for name, m in list(sys.modules.items())
                if name == "app" or name.startswith("app.")]
        return flask_app, mods
    raise ValueError(target)


# ---------------- 打桩 ----------------
def make_stubs(ai_result: dict, compare_result: dict) -> dict:
    def call_vision_auto(image_path, geometry_block="", mode="standard"):
        return json.loads(json.dumps(ai_result)), "stub-vision-model"

    def run_model_compare(image_path, prompt_text, generation_settings=None,
                          user_intent=None, on_candidate=None):
        return json.loads(json.dumps(compare_result))

    def edit_image_auto(image_path, prompt_text, generation_settings=None):
        return "http://stub.local/generated.png", "wan2.7-image", ""

    def push_image_to_esp(image_ref, append=False):
        PUSH_LOG.append([str(image_ref), bool(append)])
        n = len(PUSH_LOG)
        if _PUSH_FAIL_ON[0] is not None and n >= _PUSH_FAIL_ON[0]:
            return False, f"stub push failed at {n}"
        return True, f"stub pushed {n}"

    def get_local_ip():
        return "192.168.4.2"

    def extract_geometry(image_path):
        return None  # 与 geometry.py 的 mediapipe 缺失降级路径同形

    return {
        "call_vision_auto": call_vision_auto,
        "run_model_compare": run_model_compare,
        "edit_image_auto": edit_image_auto,
        "push_image_to_esp": push_image_to_esp,
        "get_local_ip": get_local_ip,
        "extract_geometry": extract_geometry,
    }


def apply_stubs(mods, stubs: dict) -> None:
    for m in mods:
        for k, fn in stubs.items():
            if hasattr(m, k):
                setattr(m, k, fn)


def apply_redirects(mods, paths: dict, flask_app) -> None:
    for m in mods:
        for k, v in paths.items():
            if hasattr(m, k):
                setattr(m, k, v)
    flask_app.config["UPLOAD_FOLDER"] = paths["UPLOAD_FOLDER"]


def install_tripwire() -> None:
    import requests

    def _blocked(*a, **k):
        raise AssertionError("付费接口被拦截：requests 被调用了")

    for name in ("get", "post", "put", "delete", "patch", "request"):
        setattr(requests, name, _blocked)
    requests.Session.request = _blocked


# ---------------- 计数（open / subprocess） ----------------
class Counters:
    def __init__(self):
        self.open_calls = 0
        self.api_key_opens = 0
        self.subprocess_calls = 0

    def open_w(self, real):
        def w(*a, **k):
            self.open_calls += 1
            try:
                name = str(a[0]) if a else str(k.get("file", ""))
            except Exception:
                name = ""
            if "api_keys" in name:
                self.api_key_opens += 1
            return real(*a, **k)
        return w

    def run_w(self, real):
        def w(*a, **k):
            self.subprocess_calls += 1
            return real(*a, **k)
        return w


@contextmanager
def counting():
    c = Counters()
    real_open, real_run = builtins.open, subprocess.run
    builtins.open = c.open_w(real_open)
    subprocess.run = c.run_w(real_run)
    try:
        yield c
    finally:
        builtins.open = real_open
        subprocess.run = real_run


# ---------------- 采集 ----------------
def key_tree(obj):
    """返回 JSON key 树（不含值）；列表记录长度 + 首元素 key 树。"""
    if isinstance(obj, dict):
        return {k: key_tree(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return {"__list_len__": len(obj),
                "__item0__": key_tree(obj[0]) if obj else None}
    return "__v__"


def capture(resp) -> dict:
    body = resp.get_data()
    out = {"status": resp.status_code,
           "content_type": (resp.content_type or ""),
           "len": len(body)}
    ct = resp.content_type or ""
    if "json" in ct:
        try:
            out["key_tree"] = key_tree(json.loads(body))
        except Exception:
            out["sha256"] = hashlib.sha256(body).hexdigest()
    else:
        out["sha256"] = hashlib.sha256(body).hexdigest()
    return out


def route_table(flask_app) -> str:
    rows = sorted(
        (r.rule, r.endpoint, tuple(sorted(r.methods - {"HEAD", "OPTIONS"})))
        for r in flask_app.url_map.iter_rules()
    )
    return "\n".join(f"{rule} | {endpoint} | {list(methods)}"
                     for rule, endpoint, methods in rows) + "\n"
