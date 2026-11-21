#!/usr/bin/env python3
"""阶段① 原样搬迁引擎（一次性脚本）。

把 testV1.0_backup_prompt.py 按映射表字节切片搬进 app/ 包：
- 分区覆盖全部行，一行不丢（行账目校验）；
- 函数体零改动；唯一例外 D10（3 个函数把 HUNYUAN_* 全局名改属性访问，单列）；
- import 由自由变量表自动生成（symtable + AST，不手写）；
- 断言：名字全部可解析、模块 import 图无环、每文件可编译、85 函数 ast/字节等价；
- 迁移表 tests/migration_table.md 脚本生成。
用法：cd P.A && PYTHONIOENCODING=utf-8 .venv/bin/python tests/split_phase1.py
"""
import ast
import builtins
import symtable
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
SRC = PA / "legacy" / "testV1.0_backup_prompt.py"
SRC_NAME = "testV1.0_backup_prompt.py"

# ---------------- 归属表：85 个顶层函数 ----------------
FUNCTION_OWNERS = {
    # config.py (8)
    "load_api_keys_file": "config", "update_api_keys_file": "config", "load_env": "config",
    "parse_bool": "config", "parse_int": "config", "parse_float": "config",
    "is_valid_ipv4": "config", "get_runtime_esp_config": "config",
    # contracts.py (11)
    "_repair_json_string": "contracts", "extract_json_from_text": "contracts",
    "safe_resp_text": "contracts", "is_busy_or_rate_limited": "contracts",
    "parse_qwen_image_edit_url": "contracts", "parse_image_url_any": "contracts",
    "parse_image_base64_any": "contracts", "parse_task_id_any": "contracts",
    "normalize_ai_result": "contracts", "normalize_user_intent": "contracts",
    "intent_option_label": "contracts",
    # prompts.py (3)
    "build_analysis_prompt": "prompts", "build_targeted_ideal_prompt": "prompts",
    "build_edit_prompt": "prompts",
    # strategy.py (4)
    "infer_automatic_goal": "strategy", "build_strategy_plan": "strategy",
    "build_guidance_text": "strategy", "build_generation_settings": "strategy",
    # vision.py (6)
    "call_vision_once": "vision", "call_vision_hunyuan_once": "vision",
    "call_vision_deepseek_once": "vision", "call_vision_zhipu_once": "vision",
    "check_vision_model_ready": "vision", "call_vision_auto": "vision",
    # quality.py (2)（偏差：放 vision.py 会与 imagegen 形成循环 import）
    "load_image_from_url_for_quality": "quality", "evaluate_image_quality": "quality",
    # imagegen/ (23)
    "image_file_to_jimeng_base64": "imagegen.jimeng",
    "build_jimeng_req_key_candidates": "imagegen.jimeng",
    "sha256_hex": "imagegen.jimeng", "hmac_sha256": "imagegen.jimeng",
    "build_volc_signed_headers": "imagegen.jimeng", "post_jimeng_signed": "imagegen.jimeng",
    "poll_jimeng_result": "imagegen.jimeng", "edit_with_jimeng_image": "imagegen.jimeng",
    "normalize_doubao_url": "imagegen.doubao", "list_doubao_model_ids": "imagegen.doubao",
    "build_doubao_model_candidates": "imagegen.doubao", "edit_with_doubao_seedream": "imagegen.doubao",
    "edit_with_dashscope_image_model": "imagegen.dashscope",
    "edit_with_wan27_image": "imagegen.dashscope",
    "edit_with_qwen_image_20": "imagegen.dashscope",
    "edit_with_qwen_image_edit": "imagegen.dashscope",
    "edit_with_zhipu_image": "imagegen.zhipu",
    "edit_with_hunyuan_image": "imagegen.hunyuan",
    "get_image_model_runner": "imagegen", "check_image_model_ready": "imagegen",
    "get_ready_models": "imagegen", "run_model_compare": "imagegen",
    "edit_image_auto": "imagegen",
    # esp.py (5)
    "parse_screen_size": "esp", "resolve_image_ref_to_bytes": "esp",
    "build_screen_jpeg": "esp", "push_image_to_esp": "esp", "build_esp_push_queue": "esp",
    # storage.py (8)
    "save_latest_result": "storage", "validate_session_id": "storage",
    "build_upload_filename": "storage", "save_session_data": "storage",
    "load_session_data": "storage", "allowed_file": "storage",
    "image_file_to_data_url": "storage", "save_base64_image_to_uploads": "storage",
    # net.py (4)
    "open_browser_later": "net", "get_local_ip": "net",
    "generate_qr_image": "net", "generate_access_qrcodes": "net",
    # routes/pages.py (5)
    "root": "routes.pages", "mobile_page": "routes.pages", "display_page": "routes.pages",
    "qrcode_page": "routes.pages", "uploaded_file": "routes.pages",
    # routes/api.py (5)
    "api_esp_config": "routes.api", "api_diagnose": "routes.api",
    "api_generate": "routes.api", "api_push_latest_to_esp": "routes.api",
    "api_latest_result": "routes.api",
    # routes/upload.py (1)
    "upload": "routes.upload",
}

# ---------------- 归属表：模块级常量 ----------------
CONFIG_CONST = {
    "BASE_DIR", "UPLOAD_FOLDER", "STATIC_FOLDER", "TEMPLATE_FOLDER", "LATEST_RESULT_JSON",
    "SESSION_FOLDER", "ALLOWED_EXTENSIONS", "MAX_CONTENT_LENGTH", "API_KEYS_FILE",
    "DASHSCOPE_API_KEY", "COMPAT_BASE_URL", "COMPAT_CHAT_URL", "MULTIMODAL_GEN_URL",
    "WAN_IMAGE_MODEL", "QWEN_IMAGE_MODEL", "QWEN_IMAGE_EDIT_MAX_MODEL",
    "ZHIPU_API_KEY", "ZHIPU_IMAGE_URL", "ZHIPU_IMAGE_MODEL", "ZHIPU_CHAT_URL",
    "ZHIPU_VISION_MODEL", "ZHIPU_REASONING_EFFORT",
    "DOUBAO_API_KEY", "DOUBAO_IMAGE_URL", "DOUBAO_IMAGE_MODEL", "DOUBAO_IMAGE_FALLBACK_MODEL",
    "DOUBAO_MODEL_LIST_URL", "DOUBAO_ENDPOINT_ID",
    "JIMENG_ACCESS_KEY_ID", "JIMENG_SECRET_ACCESS_KEY", "JIMENG_IMAGE_URL",
    "JIMENG_IMAGE_MODEL", "JIMENG_REQ_KEY",
    "HUNYUAN_CHAT_URL", "HUNYUAN_VISION_MODEL",
    "DEEPSEEK_API_KEY", "DEEPSEEK_CHAT_URL", "DEEPSEEK_VISION_MODEL",
    "ESP_SCREEN_ENABLED", "ESP_SCREEN_IP", "ESP_SCREEN_PORT", "ESP_SCREEN_ENDPOINT",
    "ESP_SCREEN_SIZE", "ESP_SCREEN_JPEG_QUALITY", "ESP_SCREEN_TIMEOUT_SEC",
}
HUNYUAN_STATE = {"HUNYUAN_API_KEY", "HUNYUAN_SECRET_ID", "HUNYUAN_SECRET_KEY"}
CONTRACTS_CONST = {
    "VISION_MODE_PRESETS", "VISION_MODELS",
    "MODEL_WAN_27", "MODEL_QWEN_IMAGE_20", "MODEL_SEEDREAM_50", "MODEL_QWEN_EDIT_MAX",
    "IMAGE_EDIT_MODELS", "MODEL_COMPARE_ENABLED_DEFAULT", "MODEL_COMPARE_LIST",
    "OPTIMIZATION_SCHEMA", "DEFAULT_USER_INTENT", "OPTIMIZATION_OPTION_MAP",
    "OUTPUT_SIZE_PRESETS",
}
STORE_OWNER = {}
for _n in CONFIG_CONST:
    STORE_OWNER[_n] = "config"
for _n in HUNYUAN_STATE:
    STORE_OWNER[_n] = "imagegen.hunyuan"
for _n in CONTRACTS_CONST:
    STORE_OWNER[_n] = "contracts"
STORE_OWNER["SESSION_ID_PATTERN"] = "storage"
STORE_OWNER["QWEN_IMAGE_NEGATIVE_PROMPT"] = "prompts"

# ---------------- D10：global 强制的最小语义保真改写 ----------------
D10_ALIAS_IMPORT = "import app.imagegen.hunyuan as _hunyuan_state"
D10_RENAMES = {
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

MODULE_DESC = {
    "config": "配置加载与运行时配置（load_env/get_runtime_esp_config/常量）",
    "contracts": "字段名常量、OPTIMIZATION_SCHEMA、返回体解析纯函数",
    "prompts": "全部提示词构建（纯字符串，内容冻结）",
    "strategy": "策略/引导文本构建",
    "vision": "视觉理解模型适配与选路",
    "quality": "图像质量评估（run_model_compare 依赖；单列以避免 vision↔imagegen 循环）",
    "esp": "ESP32 屏幕 JPEG 构建与推送",
    "storage": "sessions/latest_result/uploads 读写",
    "net": "本机 IP/二维码/浏览器启动",
    "imagegen": "图生图编排：注册表/就绪检查/并发对比/自动选择",
    "imagegen.dashscope": "阿里百炼图生图适配器",
    "imagegen.zhipu": "智谱图生图适配器",
    "imagegen.doubao": "豆包图生图适配器",
    "imagegen.jimeng": "即梦图生图适配器（含火山签名/轮询）",
    "imagegen.hunyuan": "混元图生图适配器（含 SDK 依赖与 HUNYUAN_* 全局状态）",
    "routes.pages": "页面路由",
    "routes.api": "API 路由",
    "routes.upload": "上传路由（转调 api_diagnose）",
    "server": "Flask 实例与路径常量装配",
    "__main__": "启动入口（横幅/二维码/自动开浏览器/app.run）",
}


def owner_to_path(owner: str) -> Path:
    if owner == "server":
        return PA / "app" / "server.py"
    if owner == "__main__":
        return PA / "app" / "__main__.py"
    if owner == "imagegen":
        return PA / "app" / "imagegen" / "__init__.py"
    if "." not in owner:
        return PA / "app" / f"{owner}.py"
    pkg, name = owner.split(".", 1)
    return PA / "app" / pkg / f"{name}.py"


def is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def store_names(node):
    return [n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]


def classify_top(node):
    """返回 owner key 或 'POOL'（纯 import 行，由绑定引擎重发）。"""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "POOL"
    if isinstance(node, ast.FunctionDef):
        assert node.name in FUNCTION_OWNERS, f"未映射函数 {node.name} @{node.lineno}"
        return FUNCTION_OWNERS[node.name]
    if isinstance(node, ast.Try):
        return "imagegen.hunyuan"   # tencent SDK try/except（唯一使用方）
    if isinstance(node, ast.If):
        return "__main__"
    names = store_names(node)
    owners = {STORE_OWNER[n] for n in names if n in STORE_OWNER}
    if owners:
        assert len(owners) == 1, f"跨归属赋值 @{node.lineno}: {owners}"
        return owners.pop()
    is_app_config = any(
        isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute)
        and isinstance(n.value.value, ast.Name) and n.value.value.id == "app"
        for n in ast.walk(node))
    is_makedirs = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "makedirs" for n in ast.walk(node))
    is_app_assign = any(
        isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "app" for t in n.targets)
        for n in ast.walk(node))
    if is_app_config or is_makedirs or is_app_assign:
        return "server"
    raise AssertionError(f"无法归属的顶层节点 @{node.lineno}: {ast.dump(node)[:80]}")


def bound_names_of(node):
    """段落在模块作用域绑定的名字（函数段只取函数名）。"""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {node.name}
    names = set(store_names(node))
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for a in n.names:
                names.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                names.add(a.asname or a.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
    return names


def load_names_scoped(node):
    """模块级段落引用的名字（不下钻嵌套函数/类体；装饰器/默认值/注解照收）。"""
    out = set()

    def walk(n):
        if isinstance(n, ast.ClassDef):
            for d in getattr(n, "decorator_list", []):
                walk(d)
            for b in n.bases:
                walk(b)
            for kw in n.keywords:
                walk(kw)
            return  # 类体的引用走 symtable
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for d in getattr(n, "decorator_list", []):
                walk(d)
            walk(n.args)
            if getattr(n, "returns", None):
                walk(n.returns)
            return  # 函数体的引用走 symtable
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
        for c in ast.iter_child_nodes(n):
            walk(c)

    walk(node)
    return out


def func_global_refs(fn_node, table):
    """函数（含嵌套块）以模块级名字解析的引用 + 装饰器/默认值/注解引用。"""
    refs = set()

    def collect(t):
        for s in t.get_symbols():
            if s.is_referenced() and s.is_global():
                refs.add(s.get_name())
        for c in t.get_children():
            collect(c)

    collect(table)
    refs |= load_names_scoped(fn_node)
    return refs


def main():
    raw = SRC.open(encoding="utf-8", newline="").read()
    lines = raw.splitlines(keepends=True)

    # D10 改写文本（仅 3 个函数体内、按函数作用域替换，锚点唯一断言）
    d10_lines = list(lines)
    for old_fn, renames in D10_RENAMES.items():
        # 定位该函数的行区间
        probe = ast.parse(raw)
        fn = next(n for n in probe.body
                  if isinstance(n, ast.FunctionDef) and n.name == old_fn)
        s, e = fn.lineno - 1, fn.end_lineno
        seg = "".join(d10_lines[s:e])
        for old, new in renames:
            assert seg.count(old) == 1, f"D10 锚点非唯一({old_fn}): {old[:40]}"
            seg = seg.replace(old, new)
        d10_lines[s:e] = seg.splitlines(keepends=True)
    transformed = "".join(d10_lines)

    tree = ast.parse(transformed)          # 分区/归属/引用按改写后文本
    tree_raw = ast.parse(raw)              # 零改动校验对照原文
    st = symtable.symtable(transformed, SRC_NAME, "exec")
    top_syms = {c.get_name(): c for c in st.get_children()}

    # ---- 分区（覆盖全部行）----
    segments, module_segs, pool_nodes, top_funcs = [], {}, [], []
    seg_of_node = {}
    prev_end = 0
    for node in tree.body:
        owner = classify_top(node)
        start, end = prev_end + 1, node.end_lineno
        prev_end = end
        segments.append((owner, start, end))
        seg_of_node[id(node)] = (owner, start, end)
        if owner == "POOL":
            pool_nodes.append((node, start, end))
            continue
        module_segs.setdefault(owner, []).append((start, end))
        if isinstance(node, ast.FunctionDef):
            top_funcs.append(node)
    assert prev_end == len(lines), f"行数不符: {prev_end} vs {len(lines)}"
    assert len(top_funcs) == 85, f"顶层函数数 {len(top_funcs)} != 85"

    # ---- 绑定/引用 ----
    module_bound, global_defs = {}, {}
    for node in tree.body:
        owner = classify_top(node)
        if owner == "POOL":
            continue
        for n in bound_names_of(node):
            if is_dunder(n):
                continue
            assert n not in global_defs or global_defs[n] == owner, \
                f"名字多归属: {n} -> {global_defs.get(n)} / {owner}"
            global_defs[n] = owner
            module_bound.setdefault(owner, set()).add(n)

    module_refs = {}
    for fn in top_funcs:
        owner = FUNCTION_OWNERS[fn.name]
        table = top_syms[fn.name]
        module_refs.setdefault(owner, set()).update(func_global_refs(fn, table))
    for node in tree.body:
        owner = classify_top(node)
        if owner == "POOL" or isinstance(node, ast.FunctionDef):
            continue
        module_refs.setdefault(owner, set()).update(load_names_scoped(node))

    pool_binding = {}
    for node, s, e in pool_nodes:
        seg = "".join(d10_lines[s - 1:e])
        if isinstance(node, ast.Import):
            for a in node.names:
                pool_binding[a.asname or a.name.split(".")[0]] = (node, seg)
        else:
            for a in node.names:
                pool_binding[a.asname or a.name] = (node, seg)

    BUILTIN = set(dir(builtins))

    # ---- 解析 + 生成 import ----
    imports_for, unresolved = {}, []
    for owner, refs in sorted(module_refs.items()):
        own = module_bound.get(owner, set())
        pool_used, cross_used, alias_used = [], set(), False
        seen_nodes = set()
        for name in sorted(refs):
            if is_dunder(name) or name in own or name in BUILTIN:
                continue
            if name == "_hunyuan_state":
                alias_used = True
                continue
            if name in pool_binding:
                node, seg = pool_binding[name]
                if id(node) not in seen_nodes:
                    seen_nodes.add(id(node))
                    pool_used.append(seg.strip("\r\n"))
                continue
            if name in global_defs:
                cross_used.add((global_defs[name], name))
                continue
            unresolved.append((owner, name))
        out = []
        if alias_used:
            out.append(D10_ALIAS_IMPORT)
        out += pool_used
        by_owner = {}
        for o, n in cross_used:
            by_owner.setdefault(o, set()).add(n)
        for o in sorted(by_owner):
            out.append(f"from app.{o} import " + ", ".join(sorted(by_owner[o])))
        imports_for[owner] = out
    assert not unresolved, f"无法解析的名字: {unresolved}"

    # ---- 循环 import 断言 ----
    graph = {}
    for owner, refs in module_refs.items():
        deps = set()
        for name in refs:
            if name == "_hunyuan_state":
                deps.add("imagegen.hunyuan")
            elif name in global_defs:
                deps.add(global_defs[name])
        graph[owner] = {d for d in deps if d != owner}
    visiting, done = set(), set()

    def dfs(n):
        if n in done:
            return
        assert n not in visiting, f"循环 import: {n}"
        visiting.add(n)
        for d in graph.get(n, ()):
            dfs(d)
        visiting.discard(n)
        done.add(n)

    for n in sorted(graph):
        dfs(n)

    # ---- 组装并写盘 ----
    if (PA / "app").exists():
        raise SystemExit("app/ 已存在，拒绝覆盖（如需重跑请先移走）")

    def body_of(owner):
        out = []
        for start, end in module_segs[owner]:
            out.append("".join(d10_lines[start - 1:end]))
        return "".join(out)

    written = {}
    for owner in sorted(module_segs, key=lambda o: min(s for s, _ in module_segs[o])):
        parts = [f'"""{MODULE_DESC[owner]}。"""', "",
                 "# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；本文件逻辑零改动（D10 例外见迁移表）。"]
        for imp in imports_for.get(owner, []):
            parts.append(imp)
        parts.append("")
        text = "\r\n".join(parts) + "\r\n" + body_of(owner)
        if not text.endswith("\r\n"):
            text += "\r\n"
        path = owner_to_path(owner)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        written[owner] = (path, text)

    # server.py：装配段（原 :365-377）+ 生成 imports
    server_parts = [f'"""{MODULE_DESC["server"]}。"""', "",
                    "# 阶段① 原样搬迁自 testV1.0_backup_prompt.py；逻辑零改动。"]
    server_parts += imports_for.get("server", [])
    server_parts.append("")
    server_text = "\r\n".join(server_parts) + "\r\n" + body_of("server")
    (PA / "app" / "server.py").write_bytes(server_text.encode("utf-8"))

    # __main__.py：sys.path 引导 + 生成 imports + 原启动块逐字
    main_parts = [f'"""{MODULE_DESC["__main__"]}。"""', "",
                  "import sys", "from pathlib import Path", "",
                  "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))", ""]
    main_parts += [i for i in imports_for.get("__main__", [])]
    main_parts.append("")
    main_text = "\r\n".join(main_parts) + "\r\n" + body_of("__main__")
    (PA / "app" / "__main__.py").write_bytes(main_text.encode("utf-8"))

    # 包装配
    (PA / "app" / "__init__.py").write_bytes(
        ('"""P.A 应用包。装配：server 建 Flask 实例，routes 注册路由。"""\r\n'
         "\r\n"
         "from app.server import app  # noqa: F401\r\n"
         "\r\n"
         "from app.routes import api, pages, upload  # noqa: F401,E402  # 触发路由注册\r\n"
         ).encode("utf-8"))
    (PA / "app" / "routes" / "__init__.py").write_bytes(
        '"""路由模块。"""\r\n'.encode("utf-8"))

    # ---- 校验 ----
    errors = []
    accounted = sum(e - s + 1 for _, s, e in segments)
    assert accounted == len(lines), f"行账目不符 {accounted} != {len(lines)}"

    all_files = [p for p, _ in written.values()]
    all_files += [PA / "app" / "server.py", PA / "app" / "__main__.py",
                  PA / "app" / "__init__.py", PA / "app" / "routes" / "__init__.py"]
    for p in all_files:
        try:
            compile(p.open(encoding="utf-8", newline="").read(), str(p), "exec")
        except SyntaxError as e:
            errors.append(f"编译失败 {p}: {e}")

    # 85 函数等价：非 D10 对照原文，D10 对照改写文本
    raw_fns = {n.name: n for n in tree_raw.body if isinstance(n, ast.FunctionDef)}
    d10_fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    raw_lines = lines
    rows = []
    for owner, (path, text) in written.items():
        new_tree = ast.parse(text)
        new_lines = text.splitlines(keepends=True)
        for n in new_tree.body:
            if not isinstance(n, ast.FunctionDef):
                continue
            is_d10 = n.name in D10_RENAMES
            o = d10_fns[n.name] if is_d10 else raw_fns[n.name]
            o_lines = d10_lines if is_d10 else raw_lines
            dec = min([d.lineno for d in n.decorator_list], default=n.lineno)
            dec_o = min([d.lineno for d in o.decorator_list], default=o.lineno)
            new_src = "".join(new_lines[dec - 1:n.end_lineno])
            old_src = "".join(o_lines[dec_o - 1:o.end_lineno])
            ok_ast = ast.dump(n) == ast.dump(o)
            ok_bytes = new_src == old_src
            rows.append((o.lineno, n.name, path.name, n.lineno,
                         "OK" if (ok_ast and ok_bytes) else f"FAIL(a={ok_ast},b={ok_bytes})",
                         "D10改写" if is_d10 else ""))
            if not (ok_ast and ok_bytes):
                errors.append(f"函数不等价: {n.name}")
    rows.sort()
    assert len(rows) == 85, f"比对函数数 {len(rows)} != 85"

    # 迁移表
    tbl = ["# 函数迁移表（阶段①，脚本生成）", "",
           f"源：`{SRC_NAME}`（{len(lines)} 行）→ `app/` 包。共 {len(rows)} 个函数。",
           "唯一例外：D10 = `global HUNYUAN_*`（:2242）强制的属性化改写，行为等价论证见阶段①报告。",
           "",
           "| 旧行号 | 函数 | 新文件 | 新行号 | 比对 | 备注 |", "|---|---|---|---|---|---|"]
    for old_ln, name, fname, new_ln, status, mark in rows:
        tbl.append(f"| {old_ln} | `{name}` | {fname} | {new_ln} | {status} | {mark} |")
    (PA / "tests" / "migration_table.md").write_text("\n".join(tbl) + "\n", encoding="utf-8")

    print("== 阶段① 搬迁引擎汇总 ==")
    print(f"源文件行数: {len(lines)} | 分区段数: {len(segments)} | 行账目: {accounted} OK")
    print(f"顶层函数: {len(rows)}/85 | 等价: {sum(1 for r in rows if r[4] == 'OK')} | D10改写: {len(D10_RENAMES)} 个函数")
    print(f"生成模块: {len(all_files)} 个文件")
    print(f"循环 import 检查: 通过（{len(graph)} 个模块无环）")
    if errors:
        print("!! 错误:")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)
    print("全部校验通过。迁移表: tests/migration_table.md")


if __name__ == "__main__":
    main()
