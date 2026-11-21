#!/usr/bin/env python3
"""阶段② 真实服务端七端点 200 检查（提示词 §6.2 逐字要求）。

起真实 main.py → 逐个 GET 七个页面/接口 → 全部 200。
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
LABEL = sys.argv[1] if len(sys.argv) > 1 else "phase2"
ENTRY = sys.argv[2] if len(sys.argv) > 2 else "main.py"
BASE = "http://127.0.0.1:5000"

PATHS = ["/", "/mobile", "/display", "/qrcode",
         "/api/esp_config", "/api/latest_result",
         "/uploads/photo_20261114_193009_716673ba.jpg"]

env = dict(os.environ, BROWSER="true", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
log = open(PA / "tests" / f"smoke_{LABEL}.log", "wb")
proc = subprocess.Popen([sys.executable, ENTRY], cwd=str(PA), env=env,
                        stdout=log, stderr=subprocess.STDOUT)

t0 = time.perf_counter()
reachable = False
for _ in range(900):
    time.sleep(0.1)
    if proc.poll() is not None:
        break
    try:
        with urllib.request.urlopen(BASE + "/api/esp_config", timeout=1) as r:
            if r.status == 200:
                reachable = True
                break
    except Exception:
        pass
startup = round(time.perf_counter() - t0, 2)

rows, bad = [], []
try:
    for p in PATHS:
        try:
            with urllib.request.urlopen(BASE + p, timeout=10) as r:
                body = r.read()
                rows.append((p, r.status, r.headers.get("Content-Type", ""),
                             len(body), hashlib.sha256(body).hexdigest()[:12]))
                if r.status != 200:
                    bad.append(p)
        except Exception as e:
            rows.append((p, f"ERR {e}", "", 0, ""))
            bad.append(p)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()

print(f"{LABEL}: startup_to_reachable_s={startup} reachable={reachable}")
for p, st, ct, ln, h in rows:
    print(f"  {st:<4} {ln:>7}B  {h}  {p}  [{ct.split(';')[0]}]")
result = {"label": LABEL, "startup_s": startup, "reachable": reachable,
          "all_200": not bad, "bad": bad,
          "rows": [{"path": p, "status": st, "len": ln, "sha": h} for p, st, _, ln, h in rows]}
(PA / "tests" / f"endpoints_{LABEL}.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
print("=== 结论:", "七端点全部 200" if (reachable and not bad) else f"失败: {bad}")
sys.exit(0 if reachable and not bad else 1)
