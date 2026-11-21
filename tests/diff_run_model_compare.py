#!/usr/bin/env python3
"""阶段③ 差分测试：run_model_compare 重构前后行为逐场景对比。

契约快照把 run_model_compare 整个打桩了，因此快照无法证明这次拆分
没有改分支条件。这里直接加载两个实现（legacy 单体里的旧版、app/imagegen
里的新版），喂同一批假 runner，比较返回体逐字节相等。

覆盖：全成功 / 首个成功但质量打分抛错 / 全部失败 / 无可用模型 /
on_candidate 回调顺序与内容 / InternalError 文案归一化。
"""
import importlib.util
import json
import sys
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PA))

FAILED = []


def check(name, ok, detail=""):
    print(("OK   " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILED.append(name)


# ---- legacy 单体（只取 run_model_compare 及其依赖）----
spec = importlib.util.spec_from_file_location("pa_legacy_rc", PA / "legacy" / "testV1.0_backup_prompt.py")
legacy = importlib.util.module_from_spec(spec)
sys.modules["pa_legacy_rc"] = legacy
spec.loader.exec_module(legacy)

import app.imagegen as new_impl  # noqa: E402

MODELS = ["m1", "m2", "m3"]


def run_case(label, runner_map, quality_map, use_candidate=True):
    """同一场景分别跑旧实现与新实现，返回 (旧结果, 新结果, 旧回调, 新回调)。"""
    def make_patches(mod):
        """给某个实现装上假 runner 与假质量打分。"""
        def fake_get_runner(name):
            return runner_map.get(name)

        def fake_quality(model_name, url, generation_settings=None, user_intent=None):
            v = quality_map.get(model_name)
            if isinstance(v, Exception):
                raise v
            return v if v else (0.5, "ok")

        patched = []
        if hasattr(mod, "get_image_model_runner"):
            mod.get_image_model_runner = fake_get_runner
            patched.append(("get_image_model_runner", fake_get_runner))
        # 新版把 evaluate_image_quality 放在 __init__ 命名空间
        if hasattr(mod, "evaluate_image_quality"):
            mod.evaluate_image_quality = fake_quality
            patched.append(("evaluate_image_quality", fake_quality))
        return patched

    # legacy：get_image_model_runner / evaluate_image_quality 都在同一命名空间
    p1 = make_patches(legacy)
    p2 = make_patches(new_impl)
    try:
        cb1, cb2 = [], []
        if use_candidate:
            r1 = legacy.run_model_compare("img.jpg", "prompt", None, None,
                                          lambda m, i: cb1.append([m, i]))
            r2 = new_impl.run_model_compare("img.jpg", "prompt", None, None,
                                            lambda m, i: cb2.append([m, i]))
        else:
            r1 = legacy.run_model_compare("img.jpg", "prompt", None, None)
            r2 = new_impl.run_model_compare("img.jpg", "prompt", None, None)
        # 并发完成顺序不确定：回调按模型名排序后比较
        cb1s = sorted(cb1, key=lambda x: x[0])
        cb2s = sorted(cb2, key=lambda x: x[0])
        return r1, r2, cb1s, cb2s
    finally:
        for name, orig in p1 + p2:
            pass  # 下一场景会重新覆盖，无需还原


# 让两边用同一份模型列表与就绪判定
legacy.MODEL_COMPARE_LIST = list(MODELS)
new_impl.MODEL_COMPARE_LIST = list(MODELS)
legacy.get_ready_models = lambda names: list(names)
new_impl.get_ready_models = lambda names: list(names)

# --- 场景 1：三个都成功 ---
r1, r2, c1, c2 = run_case("all-ok",
                          {"m1": lambda p, t, s: "http://a/1.png",
                           "m2": lambda p, t, s: "http://a/2.png",
                           "m3": lambda p, t, s: "http://a/3.png"},
                          {})
check("全成功：返回体逐字节相等", json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True))
check("全成功：候选数相同", len(r1["model_candidates"]) == len(r2["model_candidates"]) == 3)
check("全成功：回调内容相同", c1 == c2, f"{len(c1)} vs {len(c2)}")

# --- 场景 2：首个成功者质量打分抛错 → 回退到任一成功候选 ---
r1, r2, _, _ = run_case("first-ok-quality-raises",
                        {"m1": lambda p, t, s: "http://a/1.png",
                         "m2": lambda p, t, s: "http://a/2.png",
                         "m3": lambda p, t, s: "http://a/3.png"},
                        {"m1": RuntimeError("quality boom")})
check("质量抛错：返回体逐字节相等", json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True))
check("质量抛错：主图来自成功候选", bool(r2["diagram_url"]) == bool(r1["diagram_url"]),
      f'{r1["diagram_url"]} vs {r2["diagram_url"]}')

# --- 场景 3：全部失败 ---
r1, r2, _, _ = run_case("all-fail",
                        {"m1": lambda p, t, s: (_ for _ in ()).throw(RuntimeError("e1")),
                         "m2": lambda p, t, s: (_ for _ in ()).throw(RuntimeError("e2")),
                         "m3": lambda p, t, s: ""},
                        {})
check("全失败：返回体逐字节相等", json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True))
check("全失败：错误聚合非空", bool(r2["diagram_error"]), r2["diagram_error"][:60])

# --- 场景 4：InternalError 文案归一化 ---
r1, r2, _, _ = run_case("internal-error",
                        {"m1": lambda p, t, s: (_ for _ in ()).throw(RuntimeError("InternalError")),
                         "m2": lambda p, t, s: (_ for _ in ()).throw(RuntimeError("submit algo service error")),
                         "m3": lambda p, t, s: "http://a/3.png"},
                        {})
check("InternalError：返回体逐字节相等", json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True))
errs1 = sorted(c["error"] for c in r1["model_candidates"] if c["error"])
check("InternalError：文案归一化一致",
      all("服务繁忙" in e for e in errs1), str(errs1)[:80])

# --- 场景 5：没有可用模型 ---
legacy.get_ready_models = lambda names: []
new_impl.get_ready_models = lambda names: []
r1, r2, _, _ = run_case("no-models", {}, {}, use_candidate=False)
check("无可用模型：返回体逐字节相等", json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True),
      r2["diagram_error"])

print()
print("=== 结论:", "全部通过" if not FAILED else f"{len(FAILED)} 项失败: {FAILED}")
sys.exit(1 if FAILED else 0)
