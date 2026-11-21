#!/usr/bin/env python3
"""阶段③-A 配套：把 parse_* / safe_resp_text / normalize_ai_result 等
的 import 从 app.contracts 改指向 app.parsing。

只改 import 行，不动任何函数体。
"""
import ast
import re
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
APP = PA / "app"

MOVED = {"_repair_json_string", "extract_json_from_text", "safe_resp_text",
         "is_busy_or_rate_limited", "parse_qwen_image_edit_url", "parse_image_url_any",
         "parse_image_base64_any", "parse_task_id_any", "normalize_ai_result"}

pat = re.compile(r"^from app\.contracts import (.+)$")
changed = []
for f in sorted(APP.rglob("*.py")):
    if "__pycache__" in str(f) or f.name in ("contracts.py", "parsing.py"):
        continue
    src = f.open(encoding="utf-8", newline="").read()
    lines = src.splitlines(keepends=True)
    out = []
    hit = False
    for line in lines:
        m = pat.match(line.rstrip("\r\n"))
        if not m:
            out.append(line)
            continue
        names = [n.strip() for n in m.group(1).split(",")]
        moved = [n for n in names if n in MOVED]
        stay = [n for n in names if n not in MOVED]
        if not moved:
            out.append(line)
            continue
        hit = True
        eol = "\r\n" if line.endswith("\r\n") else "\n"
        if stay:
            out.append(f"from app.contracts import {', '.join(stay)}{eol}")
        out.append(f"from app.parsing import {', '.join(moved)}{eol}")
    if hit:
        new = "".join(out)
        f.write_text(new, encoding="utf-8", newline="")
        ast.parse(new)  # 语法自检
        changed.append(f.relative_to(PA).as_posix())

print("== import 改向 ==")
for c in changed:
    print("  ", c)
print(f"共 {len(changed)} 个文件")
