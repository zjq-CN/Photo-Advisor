#!/usr/bin/env python3
"""阶段③-A：contracts.py（432 行，超 P0 400）按职责拆为 constants + parsing。

纯移动：按 AST 函数边界切片，被移动的函数体逐字节不变（不手抄）。
contracts.py 保留：VISION_MODE_PRESETS / 模型常量 / OPTIMIZATION_SCHEMA /
DEFAULT_USER_INTENT / OPTIMIZATION_OPTION_MAP / normalize_user_intent /
intent_option_label / OUTPUT_SIZE_PRESETS。
parsing.py 承接：JSON 修复与提取、URL/base64/task_id 提取、normalize_ai_result。
"""
import ast
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
APP = PA / "app"
SRC = APP / "contracts.py"

# 移到 parsing.py 的函数（顺序即输出顺序）
PARSE_FUNCS = [
    "_repair_json_string",
    "extract_json_from_text",
    "safe_resp_text",
    "is_busy_or_rate_limited",
    "parse_qwen_image_edit_url",
    "parse_image_url_any",
    "parse_image_base64_any",
    "parse_task_id_any",
    "normalize_ai_result",
]

PARSING_HEADER = '''"""模型返回体解析：JSON 修复与提取、URL/base64/task_id 提取、AI 结果归一化。

阶段③ 从 contracts.py 拆出（P0 单文件 ≤400 行）。函数体逐字节未改。
"""

from typing import Any, Dict, Optional
import json
import re
import requests


'''

CONTRACTS_HEADER = '''"""字段名常量、OPTIMIZATION_SCHEMA、意图归一化。

阶段③ 从本文件拆出解析类纯函数 → app/parsing.py（P0 单文件 ≤400 行）。
常量与意图归一化逻辑逐字节未改。
"""

from typing import Any, Dict, List

from app.config import ZHIPU_REASONING_EFFORT


'''


def read(p: Path) -> str:
    return p.open(encoding="utf-8", newline="").read()


src = read(SRC)
lines = src.splitlines(keepends=True)
tree = ast.parse(src)

spans = {}
for n in tree.body:
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        dec = min([d.lineno for d in n.decorator_list], default=n.lineno)
        spans[n.name] = (dec, n.end_lineno or n.lineno)
    elif isinstance(n, (ast.Assign, ast.AnnAssign)):
        # 记录模块级常量行号，用于判断"两块之间是否有注释/常量"
        pass

missing = [n for n in PARSE_FUNCS if n not in spans]
assert not missing, f"未找到函数: {missing}"

# --- 1. parsing.py：按源顺序拼装被移动的函数段 ---
moved = sorted((spans[n] + (n,) for n in PARSE_FUNCS))
parts = []
for start, end, name in moved:
    seg = "".join(lines[start - 1:end]).rstrip("\r\n")
    parts.append(seg)
parsing_src = PARSING_HEADER + "\n\n\n".join(parts) + "\n"
(APP / "parsing.py").write_text(parsing_src, encoding="utf-8", newline="")

# --- 2. contracts.py：删掉这些函数段（含其前面的空行/注释块） ---
drop = set()
for start, end, name in moved:
    for ln in range(start, end + 1):
        drop.add(ln)
    # 吃掉函数前的连续空行，以及紧随其上的"====="注释块（如 "第二次图像编辑提示词"）
    ln = start - 1
    while ln >= 1 and lines[ln - 1].strip() == "":
        drop.add(ln)
        ln -= 1

kept = []
for i, line in enumerate(lines, start=1):
    if i in drop:
        continue
    kept.append(line)

# 去掉文件头部原有的 import 段（1-8 行区块），换成新 header
body = "".join(kept)
# 找第一个非 docstring/import 的业务行：VISION_MODE_PRESETS 前的注释
marker = "# 视觉诊断模式（前端拍照页可选"
idx = body.find(marker)
assert idx > 0, "未找到 VISION_MODE_PRESETS 前的注释标记"
body = body[idx:]

contracts_src = CONTRACTS_HEADER + body.rstrip("\r\n") + "\n"
(APP / "contracts.py").write_text(contracts_src, encoding="utf-8", newline="")

# --- 3. 校验：逐字节等值（被移动的函数段必须在 parsing.py 中原样出现） ---
parsing_back = read(APP / "parsing.py")
broken = []
for start, end, name in moved:
    seg = "".join(lines[start - 1:end]).rstrip("\r\n")
    if seg not in parsing_back:
        broken.append(name)
assert not broken, f"移动后字节不等: {broken}"

for p in (APP / "contracts.py", APP / "parsing.py"):
    compile(read(p), str(p), "exec")

n_c = len(read(APP / "contracts.py").splitlines())
n_p = len(read(APP / "parsing.py").splitlines())
print("== 阶段③-A contracts 拆分 ==")
print(f"contracts.py: {len(lines)} -> {n_c} 行" + ("  [P0 OK]" if n_c <= 400 else "  [仍超 400]"))
print(f"parsing.py:   新建 {n_p} 行" + ("  [P0 OK]" if n_p <= 400 else "  [超 400]"))
print(f"移动函数 {len(moved)} 个，逐字节等值校验: 通过")
print("编译: 通过")
