"""结构层可视化测试：对多张真实照片运行 extract_geometry，
输出检测框、量化结果和多模态提示词块，生成 HTML 报告。

运行方式（venv）：
  python -m tests.test_structure_visual
  # 或
  python tests/test_structure_visual.py

输出：tests/structure_test_images/report.html
"""

import base64
import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from geometry import (
    build_prompt_block,
    extract_geometry,
    quantize_face_size,
    quantize_position_x,
    quantize_position_y,
    quantize_roll,
    quantize_yaw,
)

TEST_IMAGE_DIR = Path(__file__).parent / "structure_test_images"


def _image_to_base64(img_path: str) -> str:
    """读取图片并转为 base64 data URL，用于在 HTML 中内嵌显示。"""
    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    ext = os.path.splitext(img_path)[1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    return f"data:{mime.get(ext.lstrip('.'), 'image/jpeg')};base64,{b64}"


def _draw_face_boxes(img_path: str, people: List[Dict[str, Any]]) -> str:
    """在图片上画人脸检测框和关键信息，返回 base64 data URL。"""
    img = cv2.imread(img_path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    for person in people:
        cx = int(person["x_norm"] * w)
        cy = int(person["y_norm"] * h)
        fw = int(person["face_width_ratio"] * w)
        fh = int(fw * 1.3)
        x1, y1 = max(0, cx - fw // 2), max(0, cy - fh // 2)
        x2, y2 = min(w, cx + fw // 2), min(h, cy + fh // 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        label = f"P{person['index']}: {person['position_x']},{person['position_y']} {person['head_pose']}"
        cv2.putText(img, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ""
    return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}"


def _generate_html_report(results: List[Dict[str, Any]], output_path: Path) -> None:
    """生成可视化 HTML 报告。"""
    html_parts = ["""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>结构层检测可视化报告</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #f5f5f5; }
  h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
  h2 { color: #555; margin-top: 40px; }
  .card { background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 20px 0; padding: 20px; }
  .card-header { display: flex; align-items: center; gap: 8px; margin-bottom: 15px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
  .badge-detected { background: #d4edda; color: #155724; }
  .badge-none { background: #f8d7da; color: #721c24; }
  .image-row { display: flex; gap: 20px; flex-wrap: wrap; }
  .image-col { flex: 1; min-width: 300px; max-width: 500px; }
  .image-col img { width: 100%; border-radius: 4px; border: 1px solid #ddd; }
  .image-col p { font-size: 12px; color: #888; text-align: center; margin: 4px 0; }
  .details { margin-top: 15px; }
  .details table { width: 100%; border-collapse: collapse; font-size: 14px; }
  .details th, .details td { padding: 6px 10px; border: 1px solid #eee; text-align: left; }
  .details th { background: #f8f9fa; font-weight: 600; }
  .prompt-block { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 12px;
                   font-family: "Courier New", monospace; font-size: 13px; white-space: pre-wrap;
                   margin-top: 10px; max-height: 200px; overflow-y: auto; }
  .summary-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
  .summary-table th, .summary-table td { padding: 8px 12px; border: 1px solid #dee2e6; text-align: left; }
  .summary-table th { background: #007bff; color: white; }
  .summary-table tr:nth-child(even) { background: #f8f9fa; }
</style>
</head>
<body>
<h1>结构层检测可视化报告</h1>
<p>测试图片目录：<code>tests/structure_test_images/</code></p>
"""]

    # Summary table
    detected = [r for r in results if r["has_faces"]]
    none = [r for r in results if not r["has_faces"]]
    html_parts.append(f"""
<h2>汇总</h2>
<table class="summary-table">
<tr><th>指标</th><th>值</th></tr>
<tr><td>总图片数</td><td>{len(results)}</td></tr>
<tr><td>检测到人脸</td><td>{len(detected)}</td></tr>
<tr><td>未检测到人脸（返回 None）</td><td>{len(none)}</td></tr>
<tr><td>人脸位置覆盖</td><td>{', '.join(sorted(set(r["position_x"] for r in detected))) or 'N/A'}</td></tr>
<tr><td>人脸大小覆盖</td><td>{', '.join(sorted(set(r["face_size"] for r in detected))) or 'N/A'}</td></tr>
<tr><td>朝向覆盖</td><td>{', '.join(sorted(set(r["head_pose"] for r in detected))) or 'N/A'}</td></tr>
<tr><td>头部端正覆盖</td><td>{', '.join(sorted(set(r["roll_desc"] for r in detected))) or 'N/A'}</td></tr>
</table>
""")

    # Each image
    html_parts.append("<h2>逐图检测结果</h2>")
    for i, r in enumerate(results, 1):
        badge_class = "badge-detected" if r["has_faces"] else "badge-none"
        badge_text = f"检测到 {r['n_faces']} 张人脸" if r["has_faces"] else "未检测到人脸"
        html_parts.append(f"""
<div class="card">
<div class="card-header">
  <span style="font-size:18px;font-weight:bold;">#{i}</span>
  <span>{r['filename']}</span>
  <span class="badge {badge_class}">{badge_text}</span>
</div>
<div class="image-row">
  <div class="image-col"><img src="{r['original_b64']}" alt="原图"><p>原图</p></div>
  <div class="image-col"><img src="{r['annotated_b64']}" alt="标注图"><p>检测框标注</p></div>
</div>
""")
        if r["has_faces"]:
            rows = ""
            for p in r["people_detail"]:
                rows += f"""<tr>
  <td>人脸{p['index']}</td>
  <td>({p['x_norm']}, {p['y_norm']})</td>
  <td>{p['position_x']}</td><td>{p['position_y']}</td>
  <td>{p['face_width_ratio']}</td><td>{p['face_size']}</td>
  <td>{p['yaw_deg']}°</td><td>{p['head_pose']}</td>
  <td>{p['roll_deg']}°</td><td>{p['roll_desc']}</td>
  <td>{p['pitch_deg']}°</td>
</tr>"""
            html_parts.append(f"""
<div class="details">
<table>
<tr><th>人脸</th><th>中心(归一化)</th>
<th>横向</th><th>纵向</th>
<th>脸宽比</th><th>大小</th>
<th>Yaw</th><th>朝向</th>
<th>Roll</th><th>歪头</th>
<th>Pitch</th></tr>
{rows}
</table>
</div>""")
            # Prompt block
            escaped_prompt = r["prompt_block"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(f"""
<div class="details">
<p><strong>生成的多模态提示词块：</strong></p>
<div class="prompt-block">{escaped_prompt}</div>
</div>""")
        html_parts.append("</div>")

    html_parts.append("</body></html>")
    output_path.write_text("\n".join(html_parts), encoding="utf-8")


class StructureVisualTest(unittest.TestCase):
    """结构层可视化集成测试：在真实图片上运行检测，生成 HTML 报告。"""

    @classmethod
    def setUpClass(cls):
        if not TEST_IMAGE_DIR.is_dir():
            raise unittest.SkipTest(f"测试图片目录不存在：{TEST_IMAGE_DIR}")
        cls.image_files = sorted(
            f for f in TEST_IMAGE_DIR.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if len(cls.image_files) < 10:
            raise unittest.SkipTest(f"测试图片不足 10 张（当前 {len(cls.image_files)} 张）")

    def test_at_least_10_images_available(self):
        self.assertGreaterEqual(len(self.image_files), 10)

    def test_extract_geometry_returns_consistent_structure(self):
        """检测结果结构完整性：有脸时必须包含所有必需字段。"""
        required_top = {"detector", "count_note", "people", "prompt_block"}
        required_person = {
            "index", "x_norm", "y_norm", "face_width_ratio",
            "yaw_deg", "pitch_deg", "roll_deg",
            "position_x", "position_y", "face_size",
            "head_pose", "roll_desc", "description",
        }
        for img_path in self.image_files:
            with self.subTest(image=img_path.name):
                result = extract_geometry(str(img_path))
                if result is not None:
                    self.assertTrue(required_top.issubset(result.keys()), f"缺少顶层字段: {img_path.name}")
                    self.assertIsInstance(result["people"], list)
                    self.assertGreater(len(result["people"]), 0)
                    for person in result["people"]:
                        self.assertTrue(
                            required_person.issubset(person.keys()),
                            f"缺少人物字段: {img_path.name}"
                        )

    def test_quantized_values_in_expected_set(self):
        """量化结果必须在允许的集合内。"""
        valid_x = {"居左", "偏左", "居中", "偏右", "居右"}
        valid_y = {"上部", "中上", "中部", "中下", "下部"}
        valid_size = {"很小", "较小", "中等", "较大", "很大"}
        valid_yaw = {"正面", "略侧", "侧脸", "背面"}
        valid_roll = {"端正", "略歪", "明显倾斜"}
        for img_path in self.image_files:
            with self.subTest(image=img_path.name):
                result = extract_geometry(str(img_path))
                if result is None:
                    continue
                for person in result["people"]:
                    self.assertIn(person["position_x"], valid_x)
                    self.assertIn(person["position_y"], valid_y)
                    self.assertIn(person["face_size"], valid_size)
                    self.assertIn(person["head_pose"], valid_yaw)
                    self.assertIn(person["roll_desc"], valid_roll)

    def test_prompt_block_format_matches_spec(self):
        """提示词块必须符合约定格式。"""
        for img_path in self.image_files:
            with self.subTest(image=img_path.name):
                result = extract_geometry(str(img_path))
                if result is None:
                    continue
                block = result["prompt_block"]
                self.assertIn("【检测器参考数据】", block)
                self.assertIn("仅供参考", block)
                self.assertNotIn("审美判断", block)
                self.assertIn(result["count_note"], block)
                for person in result["people"]:
                    self.assertIn(person["description"], block)

    def test_face_center_within_image_bounds(self):
        """人脸中心坐标必须在 [0, 1] 范围内。"""
        for img_path in self.image_files:
            with self.subTest(image=img_path.name):
                result = extract_geometry(str(img_path))
                if result is None:
                    continue
                for person in result["people"]:
                    self.assertGreaterEqual(person["x_norm"], 0.0)
                    self.assertLessEqual(person["x_norm"], 1.0)
                    self.assertGreaterEqual(person["y_norm"], 0.0)
                    self.assertLessEqual(person["y_norm"], 1.0)

    def test_face_width_ratio_positive_and_bounded(self):
        """人脸宽度占比必须为正且不超过 1。"""
        for img_path in self.image_files:
            with self.subTest(image=img_path.name):
                result = extract_geometry(str(img_path))
                if result is None:
                    continue
                for person in result["people"]:
                    self.assertGreater(person["face_width_ratio"], 0.0)
                    self.assertLessEqual(person["face_width_ratio"], 1.0)

    def test_generate_html_report(self):
        """生成完整的 HTML 可视化报告。"""
        results = []
        for img_path in self.image_files:
            result = extract_geometry(str(img_path))
            has_faces = result is not None
            n_faces = len(result["people"]) if result else 0
            people_detail = result["people"] if result else []
            prompt_block = result["prompt_block"] if result else ""
            position_x = people_detail[0]["position_x"] if people_detail else ""
            face_size = people_detail[0]["face_size"] if people_detail else ""
            head_pose = people_detail[0]["head_pose"] if people_detail else ""
            roll_desc = people_detail[0]["roll_desc"] if people_detail else ""

            annotated_b64 = _draw_face_boxes(str(img_path), people_detail) if has_faces else _image_to_base64(str(img_path))

            results.append({
                "filename": img_path.name,
                "has_faces": has_faces,
                "n_faces": n_faces,
                "people_detail": people_detail,
                "prompt_block": prompt_block,
                "position_x": position_x,
                "face_size": face_size,
                "head_pose": head_pose,
                "roll_desc": roll_desc,
                "original_b64": _image_to_base64(str(img_path)),
                "annotated_b64": annotated_b64,
            })

        report_path = TEST_IMAGE_DIR / "report.html"
        _generate_html_report(results, report_path)
        self.assertTrue(report_path.exists(), "HTML 报告未生成")
        self.assertGreater(report_path.stat().st_size, 1000, "HTML 报告内容过少")

        # 打印摘要
        detected = [r for r in results if r["has_faces"]]
        print(f"\n{'='*60}")
        print(f"结构层可视化测试完成")
        print(f"{'='*60}")
        print(f"总图片: {len(results)}, 检测到人脸: {len(detected)}, 未检测到: {len(results) - len(detected)}")
        print(f"位置覆盖: {sorted(set(r['position_x'] for r in detected))}")
        print(f"大小覆盖: {sorted(set(r['face_size'] for r in detected))}")
        print(f"朝向覆盖: {sorted(set(r['head_pose'] for r in detected))}")
        print(f"歪头覆盖: {sorted(set(r['roll_desc'] for r in detected))}")
        print(f"报告路径: {report_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
