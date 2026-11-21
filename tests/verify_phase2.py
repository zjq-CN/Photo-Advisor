#!/usr/bin/env python3
"""阶段② 静态断言：改了什么、没改什么，逐项可查。

阶段① 的 verify_phase1.py 要求 85 函数字节全等；阶段② 有意修改 3 个配置函数，
因此这里换判据：
- 85 个函数一个不少（不丢失、不改名）；
- 与 legacy 有字节差异的函数，必须精确等于预期集合（多一个就是越界）；
- load_env 调用点数量不变；
- load_env 体内不再有 subprocess（powershell 只在 _windows_user_env 里，且是批量一次）；
- 三个配置函数的实现要点逐条断言。
"""
import ast
import re
import sys
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
LEGACY = PA / "legacy" / "testV1.0_backup_prompt.py"
FAILED = []

# 阶段② 允许改动的函数：3 个配置函数。
# 另 3 个是阶段① D10 的历史改写（HUNYUAN_* 值拷贝陷阱），迁移表已单独记录，
# 不属于阶段② 的越界——基线是"阶段①完成态"，不是 legacy。
D10_PHASE1 = {"call_vision_hunyuan_once", "check_image_model_ready", "check_vision_model_ready"}
# 阶段③ 拆分/改写的 4 个函数（拆大函数 + 推送循环收进 esp.push_queue）。
# 本脚本是跨阶段累计断言：基线段始终是 legacy 单体，白名单随阶段累加。
PHASE3_CHANGED = {"api_generate", "build_strategy_plan", "run_model_compare",
                  "api_push_latest_to_esp"}
EXPECTED_MODIFIED = (
    {"load_env", "load_api_keys_file", "update_api_keys_file"}
    | D10_PHASE1
    | PHASE3_CHANGED
)

def check(name, ok, detail=""):
    print(("OK   " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILED.append(name)


def read(p: Path) -> str:
    return p.open(encoding="utf-8", newline="").read()


def top_funcs(src: str):
    """name -> 源码段（含装饰器，不含前后空行）。"""
    tree = ast.parse(src)
    text = src
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(text, node) or ""
            out[node.name] = seg
    return out


legacy_src = read(LEGACY)
legacy_funcs = top_funcs(legacy_src)

app_funcs = {}
app_files = sorted((PA / "app").rglob("*.py"))
for f in app_files:
    for name, seg in top_funcs(read(f)).items():
        app_funcs[name] = (f.relative_to(PA).as_posix(), seg)

check("85 个函数全部在 app/ 中", len(legacy_funcs) == 85 and
      all(n in app_funcs for n in legacy_funcs),
      f"legacy={len(legacy_funcs)} app={len(app_funcs)}")

missing = [n for n in legacy_funcs if n not in app_funcs]
check("无函数丢失", not missing, str(missing))

modified = {n for n, seg in legacy_funcs.items()
            if n in app_funcs and app_funcs[n][1] != seg}
extra = sorted(modified - EXPECTED_MODIFIED)
check("字节差异函数 = 预期集合", modified == EXPECTED_MODIFIED,
      f"实际={sorted(modified)} 多出={extra}")

# load_env 调用点：47 处（含 def 行）——与改前一致，证明调用方零改动
n_legacy_calls = len(re.findall(r"load_env\(", legacy_src))
n_app_calls = sum(len(re.findall(r"load_env\(", read(f))) for f in app_files)
check("load_env 调用点数量不变", n_legacy_calls == n_app_calls,
      f"legacy={n_legacy_calls} app={n_app_calls}")

# load_env 体内不得再有子进程
load_env_seg = app_funcs["load_env"][1]
check("load_env 体内无 subprocess", "subprocess" not in load_env_seg)
check("load_env 体内无 powershell", "powershell" not in load_env_seg.lower())

# powershell 只在 _windows_user_env 里，且只出现一次（批量取回）
app_all = "\n".join(read(f) for f in app_files)
check("powershell 仅批量调用一处", app_all.count('"powershell"') == 1,
      f'count={app_all.count(chr(34) + "powershell" + chr(34))}')
check("不再按名取 Windows 变量",
      "GetEnvironmentVariable('" not in app_all and "GetEnvironmentVariable(\\'" not in app_all)
check("改为批量取回", "GetEnvironmentVariables('User')" in app_all)

# 缓存失效机制在位
check("有 mtime+size 失效信号", "st_mtime_ns" in app_all and "st_size" in app_all)
check("写文件后立即失效", "invalidate_api_keys_cache()" in app_funcs["update_api_keys_file"][1])
check("超时界限保持 2.5s", "timeout=2.5" in app_all, "原实现 timeout=2.5")

print()
print("=== 结论:", "全部通过" if not FAILED else f"{len(FAILED)} 项失败: {FAILED}")
sys.exit(1 if FAILED else 0)
