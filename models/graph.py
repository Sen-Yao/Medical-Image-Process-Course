import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.segmentation import random_walker
from skimage.filters import threshold_otsu

class GraphCutSegmentation:
    def __init__(self):
        pass

    def _prepare_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    def _initialize_mask(self, image_shape):
        # 初始化掩膜
        return np.zeros(image_shape[:2], np.uint8)

    def predict(self, images):
        # 输入形状为 B, W, H, C
        B, W, H, C = images.shape
        # 输出形状为 B, W, H
        output = np.zeros((B, W, H), dtype=np.uint8)

        for i in range(B):
            image = self._prepare_image(images[i])
            mask = self._initialize_mask(image.shape)

            # 使用 grabCut 进行图像分割
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # 这里我们假设前景是病变区域，背景是其他区域
            # 需要根据具体情况调整 rect 的位置和大小
            rect = (10, 10, W-10, H-10)  # 这里的 rect 需要根据实际情况调整
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

            # 将掩膜转换为二值图像
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            output[i] = mask2

        return output


class RandomWalkSegmentation:
    def __init__(self, beta=10):
        """
        初始化分割器。

        参数:
        ----------
        beta : float
            Random Walker算法中的平滑参数。较大的值会使分割更平滑。
        """
        self.beta = beta

    def predict(self, images):
        """
        对输入的图像张量进行病变区域分割。

        参数:
        ----------
        images : numpy.ndarray
            输入图像张量，形状为 (B, W, H, C)。

        返回:
        -------
        masks : numpy.ndarray
            分割后的二值化掩码，形状为 (B, W, H)，病变区域为1，其他区域为0。
        """
        if not isinstance(images, np.ndarray):
            raise ValueError("输入必须是一个numpy数组。")
        
        if images.ndim != 4:
            raise ValueError("输入张量的形状必须为 (B, W, H, C)。")
        
        B, W, H, C = images.shape
        masks = np.zeros((B, W, H), dtype=np.uint8)
        
        for i in range(B):
            img = images[i]
            
            # 如果是彩色图像，转换为灰度图
            if C == 3 or C == 4:
                gray = rgb2gray(img)
            else:
                gray = img.squeeze()
            
            # 使用大津方法确定阈值
            thresh = threshold_otsu(gray)
            
            # 初始化标记图，0为未标记，1为背景，2为前景
            markers = np.zeros(gray.shape, dtype=np.uint)
            markers[gray < thresh] = 1  # 假设灰度值低于阈值为背景
            markers[gray >= thresh] = 2  # 灰度值高于或等于阈值为前景（病变区域）
            
            # 应用随机游走算法进行分割
            labels = random_walker(gray, markers, beta=self.beta, mode='bf')
            
            # 生成二值化掩码，病变区域为1，其他为0
            masks[i] = (labels == 1).astype(np.uint8)
        
        return masks

