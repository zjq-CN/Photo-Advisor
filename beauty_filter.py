import cv2
import mediapipe as mp
import numpy as np

class BeautyFilter:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def process(self, image_path, save_path=None, strength=50):
        """
        对人脸进行磨皮美白，同时保留五官细节。
        :param image_path: 输入图片路径
        :param save_path: 保存路径（可选）
        :param strength: 磨皮强度 (0-100)
        :return: 处理后的cv2 image (BGR)
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. 人脸检测
        results = self.face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            print("[BeautyFilter] 未检测到人脸，跳过处理")
            if save_path:
                cv2.imwrite(save_path, img)
            return img

        landmarks = results.multi_face_landmarks[0].landmark
        
        # 2. 生成皮肤掩码 (Mask)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # MediaPipe Face Mesh 索引参考
        # 脸部轮廓: 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        # 这是一个简化的凸包，包含整张脸
        face_outline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # 获取脸部多边形点
        points = []
        for idx in face_outline:
            pt = landmarks[idx]
            points.append((int(pt.x * w), int(pt.y * h)))
        
        points = np.array(points, dtype=np.int32)
        cv2.fillConvexPoly(mask, points, 255)
        
        # 排除眼睛、眉毛、嘴巴，防止磨皮导致模糊
        # 增加额外的排除区域以避免“眼影”效果（黑眼圈磨皮后显得太假）
        # 左眼: 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7
        # 右眼: 263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249
        # 嘴唇: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375
        
        exclude_lists = [
            [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7], # 左眼
            [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249], # 右眼
            [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375], # 嘴唇
            [70, 63, 105, 66, 107, 55, 65, 52, 53, 46], # 左眉 (简化)
            [300, 293, 334, 296, 336, 285, 295, 282, 283, 276] # 右眉 (简化)
        ]
        
        # 绘制排除区域掩码
        exclude_mask = np.zeros((h, w), dtype=np.uint8)
        
        for idx_list in exclude_lists:
            ex_points = []
            for idx in idx_list:
                pt = landmarks[idx]
                ex_points.append((int(pt.x * w), int(pt.y * h)))
            ex_points = np.array(ex_points, dtype=np.int32)
            cv2.fillConvexPoly(exclude_mask, ex_points, 255)
            
        # 对排除区域进行膨胀，扩大保护范围（特别是眼睛周围）
        # 避免眼睛边缘被磨皮导致像画了眼影/眼线
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        exclude_mask = cv2.dilate(exclude_mask, kernel, iterations=1)
        
        # 从总 mask 中减去排除区域
        # mask (255) - exclude_mask (255) -> 0
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_mask))

        # 3. 图像处理
        # 双边滤波 - 保持边缘的同时平滑
        # d: 像素邻域直径
        # sigmaColor: 颜色空间标准差，越大越模糊
        # sigmaSpace: 坐标空间标准差
        
        # 转换强度到参数
        # strength 0-100 -> sigmaColor 0-150
        sigma = strength * 1.5
        d = 15 # 固定邻域大小
        
        filtered = cv2.bilateralFilter(img, d=d, sigmaColor=sigma, sigmaSpace=sigma)
        
        # 4. 融合
        # 模糊 mask 边缘，使过渡自然
        mask_blur = cv2.GaussianBlur(mask, (21, 21), 0)
        mask_norm = mask_blur.astype(float) / 255.0
        mask_norm = np.expand_dims(mask_norm, axis=2) # (h, w, 1)
        
        # 原始图像 * (1 - mask) + 滤波图像 * mask
        result = img.astype(float) * (1.0 - mask_norm) + filtered.astype(float) * mask_norm
        result = result.astype(np.uint8)
        
        if save_path:
            cv2.imwrite(save_path, result)
            
        return result

# 简单的测试入口
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        processor = BeautyFilter()
        print(f"Processing {sys.argv[1]}...")
        processor.process(sys.argv[1], "beauty_result.jpg", strength=60)
        print("Done. Saved to beauty_result.jpg")
