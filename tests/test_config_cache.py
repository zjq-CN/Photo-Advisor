#!/usr/bin/env python3
"""阶段② 专项：配置热刷新 + 缓存命中 + 优先级顺序。

必须证明四件事（缺一即判阶段②失败）：
1. 改 api_keys.json 后，不重启进程即可读到新值（现有体验："API Key 改了不用重启"）；
2. 文件未变时不再读盘；
3. 四级优先级顺序与改前逐字一致；
4. Windows 用户环境变量这一级同样被缓存（不再按名 spawn powershell）。
"""
import json
import os
import sys
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PA))

import app.config as cfg  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(("OK   " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILED.append(name)


sandbox = PA / "tests" / "_cfg_probe"
sandbox.mkdir(parents=True, exist_ok=True)
keys = sandbox / "api_keys.json"

# 不替换 _windows_user_env（缓存就在其内部，替换掉就测不到缓存），
# 只打桩 subprocess.run：返回罐头 JSON，并计数真实子进程调用次数。
win_calls = {"n": 0}
real_run = cfg.subprocess.run


class _FakeCompleted:
    stdout = json.dumps({"PRIO": "from_windows"})
    stderr = ""
    returncode = 0


def fake_run(*a, **k):
    win_calls["n"] += 1
    return _FakeCompleted()


cfg.API_KEYS_FILE = str(keys)
cfg.invalidate_api_keys_cache()
cfg.subprocess.run = fake_run

# --- 1. 首次读 + 热刷新（不重启） ---
keys.write_text(json.dumps({"PROBE_KEY": "v1"}), encoding="utf-8")
check("首次读取", cfg.load_env("PROBE_KEY") == "v1", cfg.load_env("PROBE_KEY"))

keys.write_text(json.dumps({"PROBE_KEY": "v2"}), encoding="utf-8")
check("改文件后不重启即生效", cfg.load_env("PROBE_KEY") == "v2", cfg.load_env("PROBE_KEY"))

keys.write_text(json.dumps({"PROBE_KEY": "v3", "OTHER": "x"}), encoding="utf-8")
check("增键后仍即时生效", cfg.load_env("PROBE_KEY") == "v3", cfg.load_env("PROBE_KEY"))

# --- 2. 未变时不读盘 ---
import builtins  # noqa: E402

real_open = builtins.open
hits = {"n": 0}


def counting_open(*a, **k):
    name = str(a[0]) if a else str(k.get("file", ""))
    if "api_keys" in name:
        hits["n"] += 1
    return real_open(*a, **k)


builtins.open = counting_open
try:
    for _ in range(50):
        cfg.load_env("PROBE_KEY")
finally:
    builtins.open = real_open
check("50 次调用 0 次读盘", hits["n"] == 0, f"opens={hits['n']}")

# --- 3. 优先级：文件 > 进程环境变量 > Windows 用户变量 > 默认值 ---
keys.write_text(json.dumps({"PRIO": "from_file"}), encoding="utf-8")
os.environ["PRIO"] = "from_env"
check("文件压过进程环境变量", cfg.load_env("PRIO") == "from_file", cfg.load_env("PRIO"))

keys.write_text("{}", encoding="utf-8")
check("无文件值时取进程环境变量", cfg.load_env("PRIO") == "from_env", cfg.load_env("PRIO"))

del os.environ["PRIO"]
check("再退到 Windows 用户变量", cfg.load_env("PRIO") == "from_windows", cfg.load_env("PRIO"))

check("全缺时返回默认值", cfg.load_env("ABSENT_KEY", "dflt") == "dflt",
      cfg.load_env("ABSENT_KEY", "dflt"))

# --- 4. Windows 用户变量这一级也被缓存（不再每次 spawn） ---
# 注意：不能替换 _windows_user_env 本身——缓存就在它内部。
# 只打桩 subprocess.run 返回罐头 JSON，让真实的缓存逻辑跑起来。
win_calls["n"] = 0
for _ in range(50):
    cfg.load_env("ABSENT_KEY", "dflt")
check("50 次未命中 0 次取 Windows 环境（全缓存命中）", win_calls["n"] == 0,
      f"win_env_calls={win_calls['n']}")

# --- 5. 写路径立即失效（POST /api/esp_config 依赖这条） ---
cfg.update_api_keys_file({"WRITTEN": "w1"})
check("写完立刻可读", cfg.load_env("WRITTEN") == "w1", cfg.load_env("WRITTEN"))

print()
print("=== 结论:", "全部通过" if not FAILED else f"{len(FAILED)} 项失败: {FAILED}")
sys.exit(1 if FAILED else 0)
