#!/usr/bin/env python3
"""阶段③ 现状测量：文件行数、超 60 行函数、嵌套深度、提示词点名 4 函数。"""
import ast
from pathlib import Path

PA = Path(__file__).resolve().parent.parent


def depth(node, d=0):
    best = d
    for ch in ast.iter_child_nodes(node):
        if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.If, ast.For,
                           ast.While, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith)):
            best = max(best, depth(ch, d + 1))
        else:
            best = max(best, depth(ch, d))
    return best


files = sorted(p for p in (PA / "app").rglob("*.py") if "__pycache__" not in str(p))
files.append(PA / "main.py")

print("=== 文件行数 ===")
sizes = []
for f in files:
    n = len(f.open(encoding="utf-8", newline="").read().splitlines())
    sizes.append((n, f.relative_to(PA).as_posix()))
for n, rel in sorted(sizes, reverse=True):
    print(f"{n:>5}  {rel}" + ("   <<< 超 P0 400" if n > 400 else ""))

print()
print("=== 函数 >60 行（含嵌套深度，P0: ≤60 行 / ≤3 层）===")
big = []
named = {"get_runtime_esp_config", "api_generate", "build_strategy_plan",
         "run_model_compare", "edit_with_hunyuan_image", "call_vision_auto",
         "normalize_ai_result", "build_esp_push_queue", "extract_geometry"}
found = {}
for f in files:
    src = f.open(encoding="utf-8", newline="").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for n in ast.walk(tree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        dec = min([d.lineno for d in n.decorator_list], default=n.lineno)
        end = n.end_lineno or n.lineno
        size = end - dec + 1
        dep = depth(n)
        if n.name in named:
            found[n.name] = (size, f.relative_to(PA).as_posix(), dec, dep)
        if size > 60:
            big.append((size, n.name, f.relative_to(PA).as_posix(), dec, dep))

for size, name, rel, dec, dep in sorted(big, reverse=True):
    print(f"{size:>5} 行  嵌套{dep}  {rel}:{dec}  {name}()")
print(f"合计超 60 行函数: {len(big)} 个")

print()
print("=== 提示词点名 / 关键函数 ===")
for name in ["get_runtime_esp_config", "api_generate", "build_strategy_plan", "run_model_compare"]:
    if name in found:
        size, rel, dec, dep = found[name]
        print(f"  {name:<24} {size:>4} 行  嵌套{dep}  {rel}:{dec}")
    else:
        print(f"  {name:<24} 未找到")
