"""几何结构层单元测试：分箱纯函数、提示词块组装与失败路径。

不依赖真实 mediapipe（未安装时 extract_geometry 一律返回 None，测试仍成立）。
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
GEOMETRY_PATH = PROJECT_DIR / "geometry.py"

_spec = importlib.util.spec_from_file_location("geometry", GEOMETRY_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"无法加载 geometry 模块：{GEOMETRY_PATH}")
geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geometry)


class QuantizeBoundaryTests(unittest.TestCase):
    """分箱边界：左闭右开 [下界, 上界)，末档取上界及以上。"""

    def test_position_x_boundaries(self):
        cases = [
            (0.299, "居左"),
            (0.30, "偏左"),
            (0.44, "偏左"),
            (0.45, "居中"),
            (0.549, "居中"),
            (0.55, "偏右"),
            (0.699, "偏右"),
            (0.70, "居右"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(geometry.quantize_position_x(value), expected)

    def test_position_y_boundaries(self):
        cases = [
            (0.299, "上部"),
            (0.30, "中上"),
            (0.419, "中上"),
            (0.42, "中部"),
            (0.579, "中部"),
            (0.58, "中下"),
            (0.699, "中下"),
            (0.70, "下部"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(geometry.quantize_position_y(value), expected)

    def test_face_size_boundaries(self):
        cases = [
            (0.049, "很小"),
            (0.05, "较小"),
            (0.099, "较小"),
            (0.10, "中等"),
            (0.179, "中等"),
            (0.18, "较大"),
            (0.299, "较大"),
            (0.30, "很大"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(geometry.quantize_face_size(value), expected)

    def test_yaw_boundaries(self):
        cases = [
            (0.0, "正面"),
            (9.9, "正面"),
            (10.0, "略侧"),
            (34.9, "略侧"),
            (35.0, "侧脸"),
            (69.9, "侧脸"),
            (70.0, "背面"),
            (-10.0, "略侧"),
            (-35.0, "侧脸"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(geometry.quantize_yaw(value), expected)

    def test_roll_boundaries(self):
        cases = [
            (0.0, "端正"),
            (7.9, "端正"),
            (8.0, "略歪"),
            (19.9, "略歪"),
            (20.0, "明显倾斜"),
            (-8.0, "略歪"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(geometry.quantize_roll(value), expected)


class PromptBlockTests(unittest.TestCase):
    def test_build_prompt_block_contains_count_note_and_every_person(self):
        people = [
            {"description": "人脸1：位于画面居中、中上，面朝镜头，面部占比中等，头部端正。"},
            {"description": "人脸2：位于画面偏左、中部，面部微侧，面部占比较小，头部略歪。"},
        ]
        block = geometry.build_prompt_block("画面中检测到至少 2 张人脸", people)
        self.assertIn("【检测器参考数据】", block)
        self.assertIn("画面中检测到至少 2 张人脸。", block)
        self.assertIn("仅供参考", block)
        self.assertNotIn("审美判断", block)
        for person in people:
            self.assertIn(person["description"], block)


class ExtractGeometryFailureTests(unittest.TestCase):
    def test_missing_file_returns_none(self):
        self.assertIsNone(geometry.extract_geometry("/nonexistent/photo.jpg"))

    def test_corrupt_file_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(b"this is not a real image")
            path = handle.name
        try:
            self.assertIsNone(geometry.extract_geometry(path))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_blank_image_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            path = handle.name
        try:
            cv2.imwrite(path, np.zeros((100, 100, 3), dtype=np.uint8))
            self.assertIsNone(geometry.extract_geometry(path))
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
