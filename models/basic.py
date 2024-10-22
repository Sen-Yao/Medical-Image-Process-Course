import numpy as np
import cv2

class ThresholdSegmentationModel:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def predict(self, images):
        # 确保输入是浮点数类型并归一化到 [0, 1] 范围
        images = images.astype(np.float32) / 255.0

        binary_masks = []
        
        for image in images:
            # 将 RGB 图像转换为灰度图像
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 进行阈值分割
            binary_mask = (gray_image < self.threshold).astype(int)
        
            binary_masks.append(binary_mask)

        return binary_masks