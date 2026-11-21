#!/usr/bin/env python3
"""核对 api_push_latest_to_esp 与 legacy 的差异来源。"""
import ast
import difflib
from pathlib import Path

PA = Path(__file__).resolve().parent.parent


def seg(rel: str, name: str):
    src = (PA / rel).open(encoding="utf-8", newline="").read()
    lines = src.splitlines(keepends=True)
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            dec = min([d.lineno for d in n.decorator_list], default=n.lineno)
            return "".join(lines[dec - 1:(n.end_lineno or n.lineno)])
    return None


a = seg("legacy/testV1.0_backup_prompt.py", "api_push_latest_to_esp")
b = seg("app/routes/api.py", "api_push_latest_to_esp")
print("legacy 行数:", len(a.splitlines()), " 现役行数:", len(b.splitlines()))
print("=== 差异（n=2）===")
for line in difflib.unified_diff(a.splitlines(), b.splitlines(),
                                 "legacy", "app", lineterm="", n=2):
    print(line)
