#!/usr/bin/env python3
"""重新生成函数迁移表（反映阶段①②③ 全部完成后的终态）。

§8.1 要求「旧行号 + 函数名 → 新文件 + 新行号」。阶段③ 又搬了函数
（contracts → parsing、api → generation），所以整表按当前磁盘状态重算：
- 旧行号：legacy 单体 testV1.0_backup_prompt.py
- 新文件/行号：扫描当前 app/ 实际位置
- 状态：字节等值 / D10 改写 / 阶段②改写 / 阶段③拆分
另附阶段③新增函数清单与文件行数汇总。
"""
import ast
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
LEGACY = PA / "legacy" / "testV1.0_backup_prompt.py"
APP = PA / "app"
OUT = PA / "tests" / "migration_table.md"

# 阶段②有意改写
PHASE2_MODIFIED = {"load_env", "load_api_keys_file", "update_api_keys_file"}
# 阶段① D10 改写
D10_MODIFIED = {"call_vision_hunyuan_once", "check_image_model_ready", "check_vision_model_ready"}
# 阶段③ 拆分的入口函数（主体已搬进 generation.py / 内部辅助函数）
PHASE3_SPLIT = {"api_generate", "build_strategy_plan", "run_model_compare"}
# 阶段③ 只替换内联推送循环（语义不变：stop_on_first_failure=False）
PHASE3_REWIRED = {"api_push_latest_to_esp"}
# 阶段③ 新增
PHASE3_NEW_FILES = {"app/generation.py", "app/parsing.py"}


def read(p: Path) -> str:
    return p.open(encoding="utf-8", newline="").read()


def top_funcs(src: str):
    """name -> (起始行, 结束行, 源码段)"""
    tree = ast.parse(src)
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            dec = min([d.lineno for d in n.decorator_list], default=n.lineno)
            out[n.name] = (dec, n.end_lineno or n.lineno, ast.get_source_segment(src, n) or "")
    return out


legacy_src = read(LEGACY)
legacy_funcs = top_funcs(legacy_src)

# 当前 app/ 全量函数位置
cur = {}
cur_src = {}
for f in sorted(APP.rglob("*.py")):
    if "__pycache__" in str(f):
        continue
    src = read(f)
    cur_src[f.relative_to(PA).as_posix()] = src
    for name, (s, e, seg) in top_funcs(src).items():
        cur[name] = (f.relative_to(PA).as_posix(), s, e, seg)

rows = []
missing = []
for name, (old_ln, old_end, old_seg) in legacy_funcs.items():
    if name not in cur:
        missing.append(name)
        continue
    rel, new_ln, new_end, new_seg = cur[name]
    if name in PHASE3_SPLIT:
        status, note = "阶段③拆分", f"{old_end - old_ln + 1} → {new_end - new_ln + 1} 行"
    elif name in PHASE3_REWIRED:
        status, note = "阶段③改写", "内联推送循环 → esp.push_queue(stop_on_first_failure=False)"
    elif name in PHASE2_MODIFIED:
        status, note = "阶段②改写", "配置缓存化"
    elif name in D10_MODIFIED:
        status, note = "OK", "D10改写"
    elif new_seg == old_seg:
        status, note = "OK", ""
    else:
        status, note = "差异", "需人工确认"
    rows.append((old_ln, name, rel, new_ln, status, note))

rows.sort()
assert not missing, f"app/ 中缺失函数: {missing}"

# 阶段③新增函数（当前 app/ 里存在、legacy 里没有）
new_funcs = []
for name, (rel, s, e, seg) in cur.items():
    if name in legacy_funcs:
        continue
    new_funcs.append((rel, s, name, e - s + 1))

# 文件行数
sizes = []
for f in sorted(APP.rglob("*.py")):
    if "__pycache__" in str(f):
        continue
    n = len(read(f).splitlines())
    sizes.append((f.relative_to(PA).as_posix(), n))

L = []
L.append("# 函数迁移表（全阶段终态，脚本生成）")
L.append("")
L.append(f"源：`{LEGACY.name}`（{len(legacy_src.splitlines())} 行，{len(legacy_funcs)} 个顶层函数）"
         f" → `app/` 包。")
L.append("")
L.append("状态列说明：")
L.append("- `OK`：与源文件逐字节相同（阶段① 原样搬迁，引擎断言）")
L.append("- `D10改写`：`global HUNYUAN_*` 值拷贝陷阱强制的属性化改写（阶段①，行为等价论证见阶段①报告）")
L.append("- `阶段②改写`：配置读取缓存化（消灭热路径读盘与 powershell 子进程）")
L.append("- `阶段③拆分`：函数体按语义段落拆为多个短函数，入口保留原名")
L.append("- `阶段③改写`：仅把内联的重复循环替换为共享实现，分支条件与调用序不变")
L.append("")
L.append("## 一、85 个原始函数")
L.append("")
L.append("| 旧行号 | 函数 | 新文件 | 新行号 | 状态 | 备注 |")
L.append("|---|---|---|---|---|---|")
for old_ln, name, rel, new_ln, status, note in rows:
    L.append(f"| {old_ln} | `{name}` | {rel} | {new_ln} | {status} | {note} |")
L.append("")
L.append(f"合计 **{len(rows)}** 个函数，与源文件 `{len(legacy_funcs)}` 个一致。")
L.append("")
L.append("## 二、阶段③新增函数（拆分产物）")
L.append("")
L.append("| 文件 | 行号 | 函数 | 行数 |")
L.append("|---|---|---|---|")
for rel, s, name, n in sorted(new_funcs):
    L.append(f"| {rel} | {s} | `{name}` | {n} |")
L.append("")
L.append(f"合计 **{len(new_funcs)}** 个新增函数。")
L.append("")
L.append("## 三、文件行数（P0：单文件 ≤400 行）")
L.append("")
L.append("| 文件 | 行数 | P0 |")
L.append("|---|---|---|")
over = []
for rel, n in sizes:
    ok = "✅" if n <= 400 else "❌ 超限"
    if n > 400:
        over.append(rel)
    L.append(f"| {rel} | {n} | {ok} |")
L.append("")
L.append(f"合计 {len(sizes)} 个文件 / {sum(n for _, n in sizes)} 行。"
         + ("全部达标。" if not over else f"超限：{over}"))

OUT.write_text("\n".join(L) + "\n", encoding="utf-8")

print("== 迁移表已重新生成 ==")
print(f"原始函数: {len(rows)} 个（缺失 {len(missing)}）")
print(f"阶段③新增: {len(new_funcs)} 个")
print(f"文件: {len(sizes)} 个 / {sum(n for _, n in sizes)} 行 | 超 400: {over or '无'}")
by_status = {}
for *_, status, _n in rows:
    by_status[status] = by_status.get(status, 0) + 1
print("状态分布:", by_status)
