#!/usr/bin/env python3
"""§3.7 离线可启动验证：无任何 API Key 时能否起来并把"缺什么"说清楚。

难点：`import app.config` 会先执行包内 __init__，各模块在真实密钥下
都已把常量算好；之后再改 API_KEYS_FILE 不影响已算好的值。
因此这里导入后**逐模块清空各自的常量副本**，模拟"什么都没配"的真实状态。
不联网、不调任何付费接口。
"""
import os
import sys
import tempfile
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PA))

FAILED = []


def check(name, ok, detail=""):
    print(("OK   " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILED.append(name)


import app  # noqa: E402  （先完整导入，随后统一清空）
import app.config as cfg  # noqa: E402
from app.server import app as flask_app  # noqa: E402

# ---- 模拟"什么都没配" ----
KEY_NAMES = [
    "DASHSCOPE_API_KEY", "DOUBAO_API_KEY", "ZHIPU_API_KEY", "DEEPSEEK_API_KEY",
    "JIMENG_ACCESS_KEY_ID", "JIMENG_SECRET_ACCESS_KEY",
    "HUNYUAN_API_KEY", "HUNYUAN_SECRET_ID", "HUNYUAN_SECRET_KEY",
]
for k in KEY_NAMES:
    os.environ.pop(k, None)

empty = Path(tempfile.mkdtemp()) / "api_keys.json"
empty.write_text("{}", encoding="utf-8")

# config 层：指向空文件、断掉 powershell 回退、清空 ESP 配置
cfg.API_KEYS_FILE = str(empty)
cfg.invalidate_api_keys_cache()
cfg._windows_user_env = lambda: {}
cfg.ESP_SCREEN_IP = ""
cfg.ESP_SCREEN_ENABLED = True
cfg.subprocess.run = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no ipconfig"))

# 其余模块各自持有一份常量副本，逐个清空
cleared_modules = []
for name in [n for n in list(sys.modules) if n == "app" or n.startswith("app.")]:
    mod = sys.modules.get(name)
    if mod is None:
        continue
    hit = False
    for k in KEY_NAMES:
        if hasattr(mod, k):
            setattr(mod, k, "")
            hit = True
    if hasattr(mod, "ESP_SCREEN_IP"):
        setattr(mod, "ESP_SCREEN_IP", "")
        hit = True
    if hit:
        cleared_modules.append(name)

check("常量已在所有模块清空", len(cleared_modules) >= 3, f"{len(cleared_modules)} 个模块")
check("config.DASHSCOPE_API_KEY 为空串", cfg.DASHSCOPE_API_KEY == "", repr(cfg.DASHSCOPE_API_KEY))

# ---- 1) 服务仍能起：五个页面/接口全 200 ----
client = flask_app.test_client()
for path in ("/", "/mobile", "/display", "/qrcode", "/api/esp_config", "/api/latest_result"):
    r = client.get(path)
    check(f"无 Key 时 {path} 200", r.status_code == 200, str(r.status_code))

# ---- 2) 缺什么能说清楚 ----
from app.vision import check_vision_model_ready, call_vision_auto  # noqa: E402
from app.imagegen import check_image_model_ready  # noqa: E402
from app.esp import push_image_to_esp  # noqa: E402

ok, msg = check_vision_model_ready("glm-5.3-flash")
check("视觉模型缺 Key 时给出原因", (not ok) and ("未配置" in msg), msg)
ok2, msg2 = check_image_model_ready("wan2.7-image")
check("生图模型缺 Key 时给出原因", (not ok2) and ("未配置" in msg2), msg2)

try:
    call_vision_auto("nonexistent.jpg", "", mode="standard")
    check("全模型不可用时抛错", False, "没有抛错")
except RuntimeError as e:
    check("全模型不可用时抛出可读错误", "未配置可用视觉模型密钥" in str(e), str(e))
except Exception as e:
    check("全模型不可用时抛出可读错误", False, f"{type(e).__name__}: {e}")

fixture = PA / "uploads" / "photo_20261114_193009_716673ba.jpg"
ok3, msg3 = push_image_to_esp(str(fixture))
check("ESP 未配 IP 时给出原因",
      (not ok3) and ("ESP_SCREEN_IP" in msg3 or "已关闭" in msg3), msg3)

# ---- 3) 缺 Key 时 /api/diagnose 应给出可读错误而不是崩 ----
import io  # noqa: E402
img_bytes = fixture.read_bytes()
r = client.post("/api/diagnose",
                data={"photo": (io.BytesIO(img_bytes), "photo.jpg"), "mode": "standard"},
                content_type="multipart/form-data")
body = r.get_json() or {}
check("缺 Key 时 /api/diagnose 返回可读错误",
      r.status_code in (400, 500) and not body.get("ok") and bool(body.get("msg")),
      f'{r.status_code}: {str(body.get("msg"))[:70]}')

print()
print("=== 结论:", "离线可启动且错误可读" if not FAILED else f"{len(FAILED)} 项失败: {FAILED}")
sys.exit(1 if FAILED else 0)
