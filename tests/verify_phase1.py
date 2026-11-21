#!/usr/bin/env python3
"""阶段① 独立验收（不经过快照沙盒）。

核心：路径等值断言（legacy 与新树解析出的绝对路径逐字对比）——
沙盒重定向会掩盖 BASE_DIR 拆包漂移，此脚本是唯一不被掩盖的验证。
另含：85 函数字节等值、编译、load_env 计数、TENCENT 面收敛、D10 动态读、app.run 逐字。
"""
import ast
import importlib.util
import sys
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PA))
FAILED = []


def check(name, cond, detail=""):
    line = ("OK  " if cond else "FAIL") + " " + name + (" | " + detail if detail else "")
    print(line)
    if not cond:
        FAILED.append(name)


SRC = PA / "legacy" / "testV1.0_backup_prompt.py"
raw = SRC.open(encoding="utf-8", newline="").read()
raw_lines = raw.splitlines(keepends=True)

# ---------- 1. 路径等值（无沙盒，D8 雷区） ----------
spec = importlib.util.spec_from_file_location("pa_legacy", SRC)
legacy = importlib.util.module_from_spec(spec)
sys.modules["pa_legacy"] = legacy
spec.loader.exec_module(legacy)
import app.config as cfg  # noqa: E402
from app.server import app as new_app  # noqa: E402

for f in ["BASE_DIR", "UPLOAD_FOLDER", "STATIC_FOLDER", "TEMPLATE_FOLDER",
          "SESSION_FOLDER", "LATEST_RESULT_JSON", "API_KEYS_FILE"]:
    lv, nv = str(getattr(legacy, f)), str(getattr(cfg, f))
    check("路径等值 " + f, lv == nv, f"legacy={lv} new={nv}")

lt = (legacy.app.template_folder, legacy.app.static_folder)
nt = (new_app.template_folder, new_app.static_folder)
check("Flask template/static 等值", lt == nt, f"legacy={lt} new={nt}")

# ---------- 2. 编译 ----------
files = sorted((PA / "app").rglob("*.py"))
bad_compile = []
for p in files:
    try:
        compile(p.open(encoding="utf-8", newline="").read(), str(p), "exec")
    except SyntaxError as e:
        bad_compile.append(f"{p.name}:{e}")
check("编译 app/ 全树", not bad_compile, f"{len(files)} files {bad_compile}")

# ---------- 3. 85 函数字节等值 ----------
old_tree = ast.parse(raw)
old_fns = {n.name: n for n in old_tree.body
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
check("旧树顶层函数数=85", len(old_fns) == 85, str(len(old_fns)))

new_fns = {}
dups = []
for p in files:
    t = p.open(encoding="utf-8", newline="").read()
    for n in ast.parse(t).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if n.name in new_fns:
                dups.append(n.name)
            new_fns[n.name] = (p, t, n)
check("无跨文件重名", not dups, str(dups))
check("新树顶层函数数=85", len(new_fns) == 85, str(len(new_fns)))
check("函数名集合一致", set(new_fns) == set(old_fns),
      "缺:" + str(sorted(set(old_fns) - set(new_fns)))
      + " 多:" + str(sorted(set(new_fns) - set(old_fns))))

D10 = {
    "call_vision_hunyuan_once": [
        ('if HUNYUAN_API_KEY and HUNYUAN_API_KEY.startswith("AKID"):',
         'if _hunyuan_state.HUNYUAN_API_KEY and _hunyuan_state.HUNYUAN_API_KEY.startswith("AKID"):'),
        ('"Authorization": f"Bearer {HUNYUAN_API_KEY}",',
         '"Authorization": f"Bearer {_hunyuan_state.HUNYUAN_API_KEY}",'),
    ],
    "check_vision_model_ready": [
        ('return bool(HUNYUAN_API_KEY), "未配置 HUNYUAN_API_KEY"',
         'return bool(_hunyuan_state.HUNYUAN_API_KEY), "未配置 HUNYUAN_API_KEY"'),
    ],
    "check_image_model_ready": [
        ('return bool(HUNYUAN_API_KEY) or (bool(HUNYUAN_SECRET_ID) and bool(HUNYUAN_SECRET_KEY)), "未配置 HUNYUAN_API_KEY 或 SecretId/Key"',
         'return bool(_hunyuan_state.HUNYUAN_API_KEY) or (bool(_hunyuan_state.HUNYUAN_SECRET_ID) and bool(_hunyuan_state.HUNYUAN_SECRET_KEY)), "未配置 HUNYUAN_API_KEY 或 SecretId/Key"'),
    ],
}


def src_of(text, node):
    return "".join(text.splitlines(keepends=True)[node.lineno - 1: node.end_lineno])


n_ok = n_d10 = 0
for name, o in old_fns.items():
    if name not in new_fns:
        continue
    p, t, n = new_fns[name]
    o_src, n_src = src_of(raw, o), src_of(t, n)
    if name in D10:
        exp = o_src
        for a, b in D10[name]:
            assert exp.count(a) == 1, f"D10 锚点不唯一 {name}"
            exp = exp.replace(a, b)
        check("D10 改写等值 " + name, n_src == exp, f"{p.name}:{n.lineno}")
        n_d10 += 1
    else:
        check("字节等值 " + name, n_src == o_src, f"{p.name}:{n.lineno}")
        n_ok += 1
check("字节等值计数 82+D10×3", n_ok == 82 and n_d10 == 3, f"ok={n_ok} d10={n_d10}")

# ---------- 3b. 模块级块字节等值（BASE_DIR 行除外，由路径断言覆盖） ----------
all_new_text = "".join(p.open(encoding="utf-8", newline="").read() for p in files)
miss = []
for n in old_tree.body:
    if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
    seg = src_of(raw, n)
    if "BASE_DIR" in seg:
        continue  # D8 行修正，等值由路径断言证明
    if seg not in all_new_text:
        miss.append(f"L{n.lineno}-{n.end_lineno}")
check("模块级块字节等值", not miss, str(miss))

# ---------- 4. load_env 计数等值 ----------
c_old = raw.count("load_env(")
c_new = sum(p.open(encoding="utf-8", newline="").read().count("load_env(") for p in files)
check("load_env( 计数等值", c_old == c_new, f"legacy={c_old} new={c_new}")

# ---------- 5. TENCENT SDK 面收敛 ----------
bad = []
for p in files:
    if p.name == "hunyuan.py":
        continue
    t = p.open(encoding="utf-8", newline="").read()
    for kw in ("TENCENT_SDK_AVAILABLE", "hunyuan_client", "hunyuan_models",
               "ClientProfile", "HttpProfile"):
        if kw in t:
            bad.append(f"{p.name}:{kw}")
check("TENCENT 面仅在 imagegen/hunyuan.py", not bad, str(bad))

# ---------- 6. D10 动态读（值拷贝陷阱探针） ----------
from app.imagegen import check_image_model_ready  # noqa: E402
import app.imagegen.hunyuan as hy  # noqa: E402
hy.HUNYUAN_API_KEY = ""
hy.HUNYUAN_SECRET_ID = ""
hy.HUNYUAN_SECRET_KEY = ""
r0 = check_image_model_ready("hunyuan-dit")
hy.HUNYUAN_API_KEY = "AKIDtest"
hy.HUNYUAN_SECRET_KEY = "sk"
r1 = check_image_model_ready("hunyuan-dit")
check("D10 动态读 空→False", r0[0] is False, str(r0))
check("D10 动态读 设值→True", r1[0] is True, str(r1))

# ---------- 7. app.run 逐字 ----------
run_old = [l for l in raw_lines if "app.run(" in l]
run_new = [l for l in (PA / "app" / "__main__.py").open(encoding="utf-8", newline="").read().splitlines(keepends=True)
           if "app.run(" in l]
check("app.run 行逐字一致", run_old == run_new, f"{run_old} vs {run_new}")

print()
print("=== 结论:", "全部通过" if not FAILED else f"{len(FAILED)} 项失败: {FAILED}")
sys.exit(1 if FAILED else 0)
