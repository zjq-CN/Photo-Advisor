#!/usr/bin/env python3
"""阶段③-B：把 api_generate / api_push_latest_to_esp 的函数体改为调共享实现。

- api_generate：原 236 行 → 编排调用（校验→生图→推送→落盘→响应）
- api_push_latest_to_esp：内联推送循环 → esp.push_queue(stop_on_first_failure=False)
只替换函数体，路由装饰器、URL、方法、异常分支逐字保留。
"""
import ast
from pathlib import Path

PA = Path(__file__).resolve().parent.parent
API = PA / "app" / "routes" / "api.py"

NEW_API_GENERATE = '''@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        payload = request.get_json(silent=True) or {}
        err, ctx = prepare_generate_request(payload)
        if err is not None:
            return err

        print("[开始图像编辑生成理想拍摄结果图]")
        result = run_generation_models(ctx)
        esp = push_generated(ctx, result)

        apply_to_session(ctx, result, esp)
        save_session_data(ctx["session_id"], ctx["session_data"])

        save_latest_result(build_latest_payload(ctx, result, esp))
        return jsonify(build_response(ctx, result, esp))

    except FileNotFoundError as e:
        return jsonify({"ok": False, "msg": str(e)}), 404

    except requests.HTTPError as e:
        err_text = ""
        try:
            err_text = e.response.text
        except Exception:
            err_text = str(e)
        return jsonify({"ok": False, "msg": f"模型接口调用失败：{err_text}"}), 500

    except Exception as e:
        return jsonify({"ok": False, "msg": f"处理失败：{str(e)}"}), 500
'''

OLD_PUSH_BLOCK = '''        success_count = 0
        details: List[str] = []
        for i, ref in enumerate(queue):
            ok, msg = push_image_to_esp(ref, append=i > 0)
            details.append(msg)
            if ok:
                success_count += 1
        ok = success_count > 0
        msg = f"ESP push {success_count}/{len(queue)}. {(details[-1] if details else '')}".strip()
'''

NEW_PUSH_BLOCK = '''        ok, msg, success_count, _total = push_queue(
            queue, stop_on_first_failure=False, add_skip_note=False)
'''


def read(p: Path) -> str:
    return p.open(encoding="utf-8", newline="").read()


src = read(API)
lines = src.splitlines(keepends=True)
nl = "\r\n" if "\r\n" in src else "\n"

tree = ast.parse(src)
target = None
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name == "api_generate":
        target = n
        break
assert target is not None, "未找到 api_generate"

dec = min([d.lineno for d in target.decorator_list], default=target.lineno)
start, end = dec, (target.end_lineno or target.lineno)

new_body = NEW_API_GENERATE.replace("\n", nl)
out = "".join(lines[:start - 1]) + new_body + "".join(lines[end:])

# --- 手动推送循环 → 共享实现 ---
assert OLD_PUSH_BLOCK.replace("\n", nl) in out, "未找到手动推送循环原文"
out = out.replace(OLD_PUSH_BLOCK.replace("\n", nl), NEW_PUSH_BLOCK.replace("\n", nl))

# --- import：补 generation 与 push_queue，去掉不再使用的名字 ---
out = out.replace(
    "from app.esp import build_esp_push_queue, push_image_to_esp" + nl,
    "from app.esp import build_esp_push_queue, push_image_to_esp, push_queue" + nl,
)
if "from app.generation import" not in out:
    anchor = "from app.imagegen import edit_image_auto, run_model_compare" + nl
    gen_import = (
        "from app.generation import (" + nl
        + "    apply_to_session, build_latest_payload, build_response," + nl
        + "    prepare_generate_request, push_generated, run_generation_models," + nl
        + ")" + nl
    )
    assert anchor in out, "未找到 imagegen import 锚点"
    out = out.replace(anchor, anchor + gen_import)

ast.parse(out)
API.write_text(out, encoding="utf-8", newline="")

n = len(out.splitlines())
print("== 阶段③-B api.py ==")
print(f"lines: {len(lines)} -> {n}" + ("  [P0 OK]" if n <= 400 else "  [仍超 400]"))
print("ast 解析: 通过")
