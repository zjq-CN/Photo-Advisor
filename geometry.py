"""几何结构层：本地 MediaPipe 人脸检测，为视觉诊断提供低影响力参考数据。

检测失败（0 人 / 图片不可读 / 任何异常 / 未安装 mediapipe）时返回 None，
调用方应完整跳过结构层，保持原有行为不变。
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    # 未安装或依赖版本冲突都按"不可用"处理，应用与测试在无该依赖时也能跑
    mp = None
    MEDIAPIPE_AVAILABLE = False


# 分箱边界：左闭右开 [下界, 上界)，末档取上界及以上
def quantize_position_x(x: float) -> str:
    if x < 0.30:
        return "居左"
    if x < 0.45:
        return "偏左"
    if x < 0.55:
        return "居中"
    if x < 0.70:
        return "偏右"
    return "居右"


def quantize_position_y(y: float) -> str:
    if y < 0.30:
        return "上部"
    if y < 0.42:
        return "中上"
    if y < 0.58:
        return "中部"
    if y < 0.70:
        return "中下"
    return "下部"


def quantize_face_size(ratio: float) -> str:
    if ratio < 0.05:
        return "很小"
    if ratio < 0.10:
        return "较小"
    if ratio < 0.18:
        return "中等"
    if ratio < 0.30:
        return "较大"
    return "很大"


def quantize_yaw(yaw_deg: float) -> str:
    value = abs(yaw_deg)
    if value < 10:
        return "正面"
    if value < 35:
        return "略侧"
    if value < 70:
        return "侧脸"
    return "背面"


def quantize_roll(roll_deg: float) -> str:
    value = abs(roll_deg)
    if value < 8:
        return "端正"
    if value < 20:
        return "略歪"
    return "明显倾斜"


# 头部姿态关键点（MediaPipe FaceMesh 索引，以图像坐标为准）
# 鼻尖 1、下巴 152、图像左侧眼外角 33、图像右侧眼外角 263、图像左侧嘴角 61、图像右侧嘴角 291
_POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]
# 3D 参考脸（相机坐标：x 向右、y 向下、z 远离相机，鼻尖为原点）。
# 符号组合经真实照片实测：正面时 yaw/pitch 接近 0。
_REFERENCE_FACE_POINTS = np.array([
    [0.0, 0.0, 0.0],        # 鼻尖
    [0.0, 63.6, 12.5],      # 下巴
    [-43.3, -32.7, 26.0],   # 图像左侧眼外角
    [43.3, -32.7, 26.0],    # 图像右侧眼外角
    [-28.9, 28.9, 24.1],    # 图像左侧嘴角
    [28.9, 28.9, 24.1],     # 图像右侧嘴角
], dtype=np.float64)


def _rotation_to_euler(rvec: np.ndarray) -> Tuple[float, float, float]:
    """旋转向量 -> (yaw, pitch, roll) 度。

    矩阵求得的 roll 在中等 yaw 时存在 ±180° 耦合翻转，调用方不采用，
    改用 _eye_line_roll 的 2D 几何值。
    """
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.hypot(rmat[0, 0], rmat[1, 0])
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
        roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
        roll = 0.0
    return yaw, pitch, roll


def _eye_line_roll(landmarks: List[Any], width: int, height: int) -> float:
    """双眼外角连线与水平线夹角（度），由 2D 关键点直接计算，稳定可靠。"""
    left = (landmarks[33].x * width, landmarks[33].y * height)
    right = (landmarks[263].x * width, landmarks[263].y * height)
    return math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))


def _estimate_head_pose(landmarks: List[Any], width: int, height: int) -> Tuple[float, float, float]:
    """solvePnP 求 yaw/pitch，roll 用双眼连线几何值；失败时抛异常由调用方降级。"""
    image_points = np.array(
        [
            [landmarks[i].x * width, landmarks[i].y * height]
            for i in _POSE_LANDMARK_INDICES
        ],
        dtype=np.float64,
    )
    focal = float(width)
    camera_matrix = np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    success, rvec, _ = cv2.solvePnP(
        _REFERENCE_FACE_POINTS, image_points, camera_matrix, dist_coeffs
    )
    if not success:
        raise RuntimeError("solvePnP 未收敛")
    yaw, pitch, _ = _rotation_to_euler(rvec)
    return yaw, pitch, _eye_line_roll(landmarks, width, height)


def _geometric_pose(landmarks: List[Any], width: int, height: int) -> Tuple[float, float, float]:
    """solvePnP 失败时的降级近似：yaw=鼻尖相对脸中心横向偏移比例，roll=双眼连线夹角。"""
    xs = [lm.x for lm in landmarks]
    face_center_x = (min(xs) + max(xs)) / 2.0
    face_width = max(xs) - min(xs) or 1e-6
    yaw = ((landmarks[1].x - face_center_x) / face_width) * 180.0
    return yaw, 0.0, _eye_line_roll(landmarks, width, height)


def build_prompt_block(count_note: str, people: List[Dict[str, Any]]) -> str:
    """按约定格式组装检测器参考数据提示词块。"""
    lines = [
        "【检测器参考数据】",
        "以下为计算机视觉检测器非人工标注的测量结果，仅供参考，最终以图像为准；人数为\"至少\"口径（人脸被遮挡可能漏检）。",
        f"- {count_note}。",
    ]
    lines.extend(f"- {person['description']}" for person in people)
    return "\n".join(lines)


def extract_geometry(image_path: str) -> Optional[dict]:
    """整图失败（0人/不可读/异常/未安装mediapipe）返回 None，调用方应跳过结构层。"""
    if not MEDIAPIPE_AVAILABLE:
        return None
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return None
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
        ) as face_mesh:
            results = face_mesh.process(rgb_image)
        faces = results.multi_face_landmarks or []
        if not faces:
            return None

        people: List[Dict[str, Any]] = []
        for index, landmarks in enumerate(faces, start=1):
            xs = [lm.x for lm in landmarks.landmark]
            ys = [lm.y for lm in landmarks.landmark]
            center_x = (min(xs) + max(xs)) / 2.0
            center_y = (min(ys) + max(ys)) / 2.0
            face_width_ratio = max(xs) - min(xs)  # 归一化坐标下即占图宽比例
            try:
                yaw, pitch, roll = _estimate_head_pose(landmarks.landmark, width, height)
            except Exception:
                yaw, pitch, roll = _geometric_pose(landmarks.landmark, width, height)

            position_x = quantize_position_x(center_x)
            position_y = quantize_position_y(center_y)
            face_size = quantize_face_size(face_width_ratio)
            head_pose = quantize_yaw(yaw)
            roll_desc = quantize_roll(roll)
            # face_direction: head_pose 到自然语言的映射，避免"略侧朝向镜头"等不通顺表述
            face_direction = {
                "正面": "面朝镜头",
                "略侧": "面部微侧",
                "侧脸": "面部侧向",
                "背面": "背对镜头",
            }.get(head_pose, head_pose)

            people.append({
                "index": index,
                "x_norm": round(center_x, 4),
                "y_norm": round(center_y, 4),
                "face_width_ratio": round(face_width_ratio, 4),
                "yaw_deg": round(yaw, 1),
                "pitch_deg": round(pitch, 1),
                "roll_deg": round(roll, 1),
                "position_x": position_x,
                "position_y": position_y,
                "face_size": face_size,
                "head_pose": head_pose,
                "roll_desc": roll_desc,
                "description": (
                    f"人脸{index}：位于画面{position_x}、{position_y}，"
                    f"{face_direction}，面部占比{face_size}，头部{roll_desc}。"
                ),
            })

        count_note = f"画面中检测到至少 {len(people)} 张人脸"
        return {
            "detector": "mediapipe_face_mesh",
            "count_note": count_note,
            "people": people,
            "prompt_block": build_prompt_block(count_note, people),
        }
    except Exception:
        return None
