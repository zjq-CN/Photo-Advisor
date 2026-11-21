#!/usr/bin/env python3
"""真实 app.run 冒烟：启动 → 等可访问 → 探测 → 杀进程。

用法：cd P.A && .venv/bin/python tests/smoke_run.py <入口文件> <标签>
不带 shell &，子进程由本脚本托管，进程必杀。
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
entry = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else entry

log_path = Path(f"/tmp/smoke_{label}.log")
log = log_path.open("w")
env = dict(os.environ, BROWSER="true", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
t0 = time.perf_counter()
proc = subprocess.Popen([str(PA / ".venv" / "bin" / "python"), entry], cwd=str(PA),
                        stdout=log, stderr=subprocess.STDOUT, env=env)

startup_s = None
dead = None
for _ in range(600):  # 最多 60s：import（mediapipe/cv2 + 36×load_env）可能超 15s
    time.sleep(0.1)
    if proc.poll() is not None:
        dead = proc.returncode
        break
    try:
        with urllib.request.urlopen("http://127.0.0.1:5000/api/esp_config", timeout=1) as r:
            if r.status == 200:
                startup_s = round(time.perf_counter() - t0, 3)
                break
    except Exception:
        pass
if dead is not None:
    print(f"{label}: 进程提前退出 exit={dead}")
print(f"{label}: startup_to_reachable_s={startup_s} reachable={startup_s is not None}")

try:
    with urllib.request.urlopen("http://127.0.0.1:5000/", timeout=3) as r:
        print(f"{label}: GET / -> {r.status} len={len(r.read())}")
except Exception as e:
    print(f"{label}: GET / -> FAIL {e}")

proc.terminate()
try:
    proc.wait(timeout=5)
except Exception:
    proc.kill()
    proc.wait(timeout=5)
print(f"{label}: killed exit={proc.returncode}")
log.close()
print("--- log tail ---")
print("".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-8:]))
