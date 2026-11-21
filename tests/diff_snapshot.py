#!/usr/bin/env python3
"""契约快照对比：tests/baseline vs tests/<阶段>_new。

判据（硬）：路由三元组表逐字、场景状态码/content-type/key树/sha256 逐项、
push_logs（D5 分歧指纹）、perf 计数。
perf 计数模式：
  strict          改前改后必须逐项相等（阶段①原样搬迁用）
  no-regression   改后每项必须 ≤ 改前（阶段②以优化为目标，计数只许降不许升）
计时仅参考（±20% 软提示）。session_id 不可比，剔除。
"""
import json
import sys
from pathlib import Path

PERF_MODE = "strict"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
for a in sys.argv[1:]:
    if a.startswith("--perf="):
        PERF_MODE = a.split("=", 1)[1]
A = Path(args[0])
B = Path(args[1])
FAILED = []


def eq(name, x, y):
    if x != y:
        FAILED.append(name)
        print("DIFF " + name)
        print("  A:", json.dumps(x, ensure_ascii=False)[:300])
        print("  B:", json.dumps(y, ensure_ascii=False)[:300])
    else:
        print("OK   " + name)


eq("route_table 逐字",
   (A / "route_table.txt").read_text(encoding="utf-8"),
   (B / "route_table.txt").read_text(encoding="utf-8"))

sa = json.loads((A / "snapshots.json").read_text(encoding="utf-8"))
sb = json.loads((B / "snapshots.json").read_text(encoding="utf-8"))
eq("场景名集合", sorted(sa), sorted(sb))
for k in sorted(set(sa) & set(sb)):
    eq("场景 " + k, sa[k], sb[k])

va = json.loads((A / "values.json").read_text(encoding="utf-8"))
vb = json.loads((B / "values.json").read_text(encoding="utf-8"))
va.pop("diagnose_session_id", None)
vb.pop("diagnose_session_id", None)
eq("values（剔 session_id）", va, vb)

eq("push_logs（D5 指纹）",
   json.loads((A / "push_logs.json").read_text(encoding="utf-8")),
   json.loads((B / "push_logs.json").read_text(encoding="utf-8")))

pa = json.loads((A / "perf.json").read_text(encoding="utf-8"))
pb = json.loads((B / "perf.json").read_text(encoding="utf-8"))


def perf_eq(name, x, y):
    if PERF_MODE == "strict":
        eq(name, x, y)
        return
    # no-regression：每一项都不许比改前更大（负值/缺失按 0 处理）
    bad = {k: (x.get(k, 0), y.get(k, 0)) for k in set(x) | set(y)
           if y.get(k, 0) > x.get(k, 0)}
    if bad:
        FAILED.append(name)
        print("DIFF " + name + "（贪改后变大）")
        for k, (a, b) in sorted(bad.items()):
            print(f"  {k}: {a} -> {b}")
    else:
        delta = " | ".join(f"{k}: {x.get(k,0)}->{y.get(k,0)}" for k in sorted(x))
        print(f"OK   {name} [{delta}]")


perf_eq("perf 计数 normal", pa.get("normal_request", {}), pb.get("normal_request", {}))
perf_eq("perf 计数 worst", pa.get("worst_case_request", {}), pb.get("worst_case_request", {}))
ta = pa.get("esp_config_x20_ms", 0)
tb = pb.get("esp_config_x20_ms", 0)
ratio = (tb / ta) if ta else 0
print(f"INFO esp_config_x20_ms: A={ta} B={tb} ratio={ratio:.2f}（±20% 内参考通过，计数已硬对）")
if ta and not (0.8 <= ratio <= 1.2):
    print("WARN 计时超 ±20%（仅参考项）")

print()
print("=== 对比结论:", "全部一致" if not FAILED else f"{len(FAILED)} 处不一致: {FAILED}")
sys.exit(1 if FAILED else 0)
