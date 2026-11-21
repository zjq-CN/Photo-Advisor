#!/usr/bin/env python3
"""§9 测量：加第 8 个图生图模型 / 改一个模型适配器需要动几处。

不猜，逐项数出来：
- 模型名字符串常量
- 注册表条目
- 就绪检查分支
- 适配器函数本体
"""
import ast
import re
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
APP = PA / "app"


def read(p: Path) -> str:
    return p.open(encoding="utf-8", newline="").read()


print("=== 1) 现有 3 个在册模型的引用点 ===")
model = "wan2.7-image"
hits = []
for f in sorted(APP.rglob("*.py")):
    if "__pycache__" in str(f):
        continue
    for i, line in enumerate(read(f).splitlines(), 1):
        if model in line:
            hits.append((f.relative_to(PA).as_posix(), i, line.strip()[:88]))
for rel, i, line in hits:
    print(f"  {rel}:{i}  {line}")
print(f"  → 共 {len(hits)} 处，分布在 {len({h[0] for h in hits})} 个文件")

print()
print("=== 2) 注册表与就绪检查（新增模型必须改的两处） ===")
ig = read(APP / "imagegen" / "__init__.py")
tree = ast.parse(ig)
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name in ("get_image_model_runner", "check_image_model_ready"):
        lines = ig.splitlines()
        print(f"  --- {n.name}  ({'imagegen/__init__.py'}:{n.lineno}) ---")
        for ln in range(n.lineno, (n.end_lineno or n.lineno) + 1):
            print("   ", lines[ln - 1].rstrip()[:96])

print()
print("=== 3) 适配器函数本体（每个模型一个文件） ===")
for f in sorted((APP / "imagegen").glob("*.py")):
    if f.name == "__init__.py":
        continue
    src = read(f)
    fns = [n.name for n in ast.parse(src).body
           if isinstance(n, ast.FunctionDef) and n.name.startswith("edit_with_")]
    print(f"  {f.name:<16} {len(src.splitlines()):>4} 行   {', '.join(fns) or '(无 edit_with_*)'}")

print()
print("=== 4) contracts.py 里的模型常量块 ===")
c = read(APP / "contracts.py").splitlines()
for i, line in enumerate(c, 1):
    if re.match(r"^MODEL_|^IMAGE_EDIT_MODELS|^MODEL_COMPARE", line):
        print(f"  contracts.py:{i}  {line.rstrip()[:88]}")
