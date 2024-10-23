import numpy as np
import cv2
from skimage.segmentation import active_contour

class ActiveContourSegmentation:
    def __init__(self, alpha=0.015, beta=10, gamma=0.001):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def predict(self, images):
        """
        对输入的批量图像应用主动轮廓分割。

        参数:
        images: np.ndarray, 形状为 (B, W, H, C) 的批量图像

        返回:
        np.ndarray, 形状为 (B, W, H) 的分割结果
        """
        B, W, H, C = images.shape
        results = np.zeros((B, W, H), dtype=np.uint8)

        for i in range(B):
            # 选择当前图像并转换为灰度
            image = images[i]
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # 初始化轮廓
            s = np.linspace(0, 2 * np.pi, 400)
            x = W // 2 + (W // 4) * np.cos(s)  # 中心点
            y = H // 2 + (H // 4) * np.sin(s)  # 中心点
            initial = np.array([x, y]).T

            # 应用主动轮廓模型
            snake = active_contour(gray_image, initial, alpha=self.alpha, beta=self.beta, gamma=self.gamma)

            # 创建分割结果
            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [snake.astype(np.int32)], 1)

            # 将结果存储到输出数组
            results[i] = mask

        return results