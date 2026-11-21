#!/usr/bin/env python3
"""阶段③-C：拆 build_strategy_plan（128 行）与 run_model_compare（128 行）。

纪律：分支条件一行不改。
- build_strategy_plan：四张纯字面量对照表提到模块级（源码原文提取，不手抄），
  主函数只剩组装步骤，if 判断原样搬动。
- run_model_compare：把"单个模型的一次尝试"与"首个成功者标记"抽成函数，
  标记时机保持在 URL 校验之后、质量打分之前（与原实现同序）。
"""
import ast
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
STRATEGY = PA / "app" / "strategy.py"
IMAGEGEN = PA / "app" / "imagegen" / "__init__.py"


def read(p: Path) -> str:
    return p.open(encoding="utf-8", newline="").read()


def nl_of(src: str) -> str:
    return "\r\n" if "\r\n" in src else "\n"


def span_of(node):
    dec = min([d.lineno for d in node.decorator_list], default=node.lineno)
    return dec, (node.end_lineno or node.lineno)


# ======================= strategy.py =======================
src = read(STRATEGY)
NL = nl_of(src)
tree = ast.parse(src)

fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_strategy_plan")

# 从源码原文提取四张表与基础 strategy 字面量
tables = {}
base_literal = None
for node in ast.walk(fn):
    # goal_updates / shot_updates 带类型注解 → AnnAssign；另两张是普通 Assign
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
            and node.value is not None:
        name = node.target.id
        if name in ("goal_updates", "shot_updates", "mood_updates", "strength_updates"):
            tables[name] = ast.get_source_segment(src, node.value)
        if name == "strategy" and isinstance(node.value, ast.Dict):
            base_literal = ast.get_source_segment(src, node.value)
    elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name in ("goal_updates", "shot_updates", "mood_updates", "strength_updates"):
            tables[name] = ast.get_source_segment(src, node.value)
        if name == "strategy" and isinstance(node.value, ast.Dict):
            base_literal = ast.get_source_segment(src, node.value)

assert set(tables) == {"goal_updates", "shot_updates", "mood_updates", "strength_updates"}, tables
assert base_literal, "未提取到基础 strategy 字面量"

CONSTS = (
    "GOAL_UPDATES: Dict[str, Dict[str, str]] = " + tables["goal_updates"] + NL + NL + NL
    + "SHOT_UPDATES: Dict[str, Dict[str, str]] = " + tables["shot_updates"] + NL + NL + NL
    + "MOOD_UPDATES: Dict[str, str] = " + tables["mood_updates"] + NL + NL + NL
    + "STRENGTH_UPDATES: Dict[str, str] = " + tables["strength_updates"] + NL + NL + NL
)

NEW_FN = (
    "def _build_base_strategy(" + NL
    + "    user_intent: Dict[str, str]," + NL
    + "    requested_goal: str," + NL
    + "    effective_goal: str," + NL
    + ") -> Dict[str, str]:" + NL
    + '    """基础策略骨架（标签来自意图选项，值与原实现逐字相同）。"""' + NL
    + "    return " + base_literal + NL + NL + NL
    + "def build_strategy_plan(ai_result: Dict[str, str], user_intent: Dict[str, str]) -> Dict[str, str]:" + NL
    + '    requested_goal = user_intent["optimization_goal"]' + NL
    + '    effective_goal = infer_automatic_goal(ai_result) if requested_goal == "auto" else requested_goal' + NL + NL
    + "    strategy = _build_base_strategy(user_intent, requested_goal, effective_goal)" + NL
    + "    strategy.update(GOAL_UPDATES[effective_goal])" + NL
    + '    strategy.update(SHOT_UPDATES[user_intent["shot_type"]])' + NL + NL
    + '    mood_strategy = MOOD_UPDATES[user_intent["mood_style"]]' + NL
    + '    if effective_goal == "lighting":' + NL
    + "        # 光线主目标需要保留白平衡/肤色修复要求；氛围只决定调色方向，" + NL
    + "        # 不能像旧版那样把光线策略整体覆盖掉。" + NL
    + "        strategy[\"color_strategy\"] = f\"{strategy['color_strategy']}；氛围控制：{mood_strategy}\"" + NL
    + "    else:" + NL
    + '        strategy["color_strategy"] = mood_strategy' + NL + NL
    + '    strategy["strength_rule"] = STRENGTH_UPDATES[user_intent["edit_strength"]]' + NL + NL
    + '    light_text = ai_result.get("light_source_inference", "")' + NL
    + '    if "不明显" in light_text or "无法" in light_text:' + NL
    + "        strategy[\"light_strategy\"] += \" 原图光源方向不可靠，不制造新的窗光、轮廓光或逆光。\"" + NL + NL
    + "    return strategy" + NL
)

lines = src.splitlines(keepends=True)
start, end = span_of(fn)
out = "".join(lines[:start - 1]) + NEW_FN + "".join(lines[end:])
# 常量块插在 build_strategy_plan 之前（即 infer_automatic_goal 之后）
marker = "def _build_base_strategy("
idx = out.find(marker)
out = out[:idx] + CONSTS + out[idx:]

ast.parse(out)
STRATEGY.write_text(out, encoding="utf-8", newline="")
n_s = len(out.splitlines())
print(f"strategy.py: {len(lines)} -> {n_s} 行")

# ======================= imagegen/__init__.py =======================
src2 = read(IMAGEGEN)
NL2 = nl_of(src2)
tree2 = ast.parse(src2)
fn2 = next(n for n in tree2.body
           if isinstance(n, ast.FunctionDef) and n.name == "run_model_compare")

HELPERS = (
    "def _valid_image_url(url: str) -> bool:" + NL2
    + "    s = str(url).strip()" + NL2
    + "    return bool(s) and (" + NL2
    + '        s.startswith("http://")' + NL2
    + '        or s.startswith("https://")' + NL2
    + '        or s.startswith("/uploads/")' + NL2
    + "    )" + NL2 + NL2 + NL2
    + "def _attempt_model(" + NL2
    + "    model_name: str," + NL2
    + "    image_path: str," + NL2
    + "    prompt_text: str," + NL2
    + "    generation_settings: Optional[Dict[str, Any]]," + NL2
    + "    user_intent: Optional[Dict[str, str]]," + NL2
    + "    mark_first_ok: Callable[[str], None]," + NL2
    + ") -> Optional[Dict[str, Any]]:" + NL2
    + '    """单个模型的一次尝试；runner 不存在时返回 None。' + NL2 + NL2
    + "    mark_first_ok 在 URL 校验通过后、质量打分之前调用——与原实现的标记时机同序。" + NL2
    + '    """' + NL2
    + "    runner = get_image_model_runner(model_name)" + NL2
    + "    if runner is None:" + NL2
    + "        return None" + NL2
    + "    try:" + NL2
    + "        raw_url = runner(image_path, prompt_text, generation_settings)" + NL2
    + "        if not _valid_image_url(raw_url):" + NL2
    + '            raise RuntimeError("模型返回空图片地址。")' + NL2
    + "        # 第一个成功者即标记为主图模型，不阻塞等待质量打分" + NL2
    + "        mark_first_ok(model_name)" + NL2
    + "        quality_score, quality_reason = evaluate_image_quality(" + NL2
    + "            model_name," + NL2
    + "            raw_url," + NL2
    + "            generation_settings=generation_settings," + NL2
    + "            user_intent=user_intent," + NL2
    + "        )" + NL2
    + "        return {" + NL2
    + '            "model": model_name,' + NL2
    + '            "raw_diagram_url": raw_url,' + NL2
    + '            "diagram_url": raw_url,' + NL2
    + '            "beauty_applied": False,' + NL2
    + '            "error": "",' + NL2
    + '            "quality_score": quality_score,' + NL2
    + '            "quality_reason": quality_reason,' + NL2
    + "        }" + NL2
    + "    except Exception as e:" + NL2
    + "        err = str(e)" + NL2
    + "        if not err and isinstance(e, requests.HTTPError):" + NL2
    + "            try:" + NL2
    + "                err = safe_resp_text(e.response)" + NL2
    + "            except Exception:" + NL2
    + '                err = ""' + NL2
    + '        if "InternalError" in err or "submit algo service error" in err:' + NL2
    + '            err = "服务繁忙（InternalError），已自动回退"' + NL2
    + "        return {" + NL2
    + '            "model": model_name,' + NL2
    + '            "raw_diagram_url": "",' + NL2
    + '            "diagram_url": "",' + NL2
    + '            "beauty_applied": False,' + NL2
    + '            "error": err,' + NL2
    + '            "quality_score": -999.0,' + NL2
    + '            "quality_reason": "",' + NL2
    + "        }" + NL2 + NL2 + NL2
)

NEW_COMPARE = (
    "def run_model_compare(" + NL2
    + "    image_path: str," + NL2
    + "    prompt_text: str," + NL2
    + "    generation_settings: Optional[Dict[str, Any]] = None," + NL2
    + "    user_intent: Optional[Dict[str, str]] = None," + NL2
    + "    on_candidate: Optional[Callable[[str, Dict[str, Any]], None]] = None," + NL2
    + ") -> Dict[str, Any]:" + NL2
    + '    """并发调用多个生图模型，谁先成功谁作为主图（不用等全部跑完再选优）。' + NL2 + NL2
    + "    返回结构保持兼容：diagram_url/image_model 为主图结果，" + NL2
    + "    model_candidates 为全部候选（按 MODEL_COMPARE_LIST 顺序），供前端对比展示。" + NL2 + NL2
    + "    on_candidate: 每完成一个模型（成功或失败）就调用一次 (model_name, item)，" + NL2
    + "    用于增量通知（例如把已完成的候选实时写回 latest.json，前端逐张显示）。" + NL2
    + '    """' + NL2
    + "    active_models = get_ready_models(MODEL_COMPARE_LIST)" + NL2
    + "    if not active_models:" + NL2
    + "        return _empty_compare_result()" + NL2 + NL2
    + "    results: Dict[str, Dict[str, Any]] = {}" + NL2
    + "    first_ok_model: List[str] = []" + NL2
    + "    lock = threading.Lock()" + NL2
    + "    first_ok_lock = threading.Lock()" + NL2 + NL2
    + "    def _mark_first_ok(model_name: str) -> None:" + NL2
    + "        with first_ok_lock:" + NL2
    + "            if not first_ok_model:" + NL2
    + "                first_ok_model.append(model_name)" + NL2 + NL2
    + "    def _worker(model_name: str) -> None:" + NL2
    + "        item = _attempt_model(model_name, image_path, prompt_text," + NL2
    + "                              generation_settings, user_intent, _mark_first_ok)" + NL2
    + "        if item is None:" + NL2
    + "            return" + NL2
    + "        with lock:" + NL2
    + "            results[model_name] = item" + NL2
    + "        if on_candidate is not None:" + NL2
    + "            try:" + NL2
    + "                on_candidate(model_name, item)" + NL2
    + "            except Exception:" + NL2
    + "                pass" + NL2 + NL2
    + "    threads = [" + NL2
    + "        threading.Thread(target=_worker, args=(m,), daemon=True)" + NL2
    + "        for m in active_models" + NL2
    + "    ]" + NL2
    + "    for t in threads:" + NL2
    + "        t.start()" + NL2
    + "    for t in threads:" + NL2
    + "        t.join()" + NL2 + NL2
    + "    return _select_compare_result(active_models, results, first_ok_model)" + NL2 + NL2 + NL2
    + "def _empty_compare_result() -> Dict[str, Any]:" + NL2
    + "    return {" + NL2
    + '        "diagram_url": "",' + NL2
    + '        "raw_diagram_url": "",' + NL2
    + '        "image_model": "",' + NL2
    + '        "diagram_error": "未配置任何可用图像模型。",' + NL2
    + '        "beauty_applied": False,' + NL2
    + '        "model_candidates": [],' + NL2
    + "    }" + NL2 + NL2 + NL2
    + "def _select_compare_result(" + NL2
    + "    active_models: List[str]," + NL2
    + "    results: Dict[str, Dict[str, Any]]," + NL2
    + "    first_ok_model: List[str]," + NL2
    + ") -> Dict[str, Any]:" + NL2
    + '    """按原实现规则挑主图：首个成功者优先，失败则回退到任一成功候选。"""' + NL2
    + '    candidates = [results[m] for m in active_models if m in results]' + NL2
    + '    chosen_model = first_ok_model[0] if first_ok_model else ""' + NL2
    + "    chosen = results.get(chosen_model) if chosen_model else None" + NL2 + NL2
    + "    # 若首个成功模型因质量检测被标记失败（极少见），回退到任一成功候选" + NL2
    + '    if chosen is None or chosen.get("error"):' + NL2
    + '        chosen = next((c for c in candidates if not c.get("error")), None)' + NL2 + NL2
    + "    if chosen:" + NL2
    + "        return {" + NL2
    + '            "diagram_url": chosen["diagram_url"],' + NL2
    + '            "raw_diagram_url": chosen["raw_diagram_url"],' + NL2
    + '            "image_model": chosen["model"],' + NL2
    + '            "diagram_error": "",' + NL2
    + '            "beauty_applied": chosen["beauty_applied"],' + NL2
    + '            "model_candidates": candidates,' + NL2
    + "        }" + NL2 + NL2
    + '    errors = [c.get("error", "") for c in candidates if c.get("error")]' + NL2
    + "    return {" + NL2
    + '        "diagram_url": "",' + NL2
    + '        "raw_diagram_url": "",' + NL2
    + '        "image_model": "",' + NL2
    + '        "diagram_error": "；".join(errors) or "所有生图模型调用失败。",' + NL2
    + '        "beauty_applied": False,' + NL2
    + '        "model_candidates": candidates,' + NL2
    + "    }" + NL2
)

lines2 = src2.splitlines(keepends=True)
s2, e2 = span_of(fn2)
out2 = "".join(lines2[:s2 - 1]) + HELPERS + NEW_COMPARE + "".join(lines2[e2:])
ast.parse(out2)
IMAGEGEN.write_text(out2, encoding="utf-8", newline="")
print(f"imagegen/__init__.py: {len(lines2)} -> {len(out2.splitlines())} 行")
print("ast 解析: 均通过")
