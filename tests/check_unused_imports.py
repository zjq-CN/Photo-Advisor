#!/usr/bin/env python3
"""阶段③ 收尾检查：拆分后残留的无用 import。

只报告：函数体里引用了哪些顶层名字，与 import 进来的名字做差集。
"""
import ast
import sys
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
TARGETS = ["app/routes/api.py", "app/generation.py", "app/strategy.py",
           "app/contracts.py", "app/parsing.py", "app/imagegen/__init__.py",
           "app/esp.py"]

for rel in TARGETS:
    p = PA / rel
    src = p.open(encoding="utf-8", newline="").read()
    tree = ast.parse(src)
    imported = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name != "*":
                    imported[a.asname or a.name] = n.lineno
        elif isinstance(n, ast.Import):
            for a in n.names:
                imported[(a.asname or a.name).split(".")[0]] = n.lineno
    used = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            used.add(n.id)
        elif isinstance(n, ast.Attribute):
            pass
    # 属性访问的根名字也要算（requests.post 等）
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            cur = n
            while isinstance(cur, ast.Attribute):
                cur = cur.value
            if isinstance(cur, ast.Name):
                used.add(cur.id)
    # 字符串注解不算，简单起见只看 Name 节点
    unused = sorted((ln, name) for name, ln in imported.items() if name not in used)
    if unused:
        print(f"--- {rel} ---")
        for ln, name in unused:
            print(f"  :{ln}  未使用: {name}")
    else:
        print(f"--- {rel} --- 无残留")
