import numpy as np
import cv2
import torch

from torchvision import models

from skimage.color import rgb2gray
from skimage.segmentation import random_walker
from skimage.filters import threshold_otsu

from feature_extraction.intensity import sobel, prewitt
from feature_extraction.texture import apply_laws_kernel

class GraphCutSegmentation:
    def __init__(self, feature):
        self.feature = feature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.feature == "CNN":
            self.model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT').to(self.device)
            self.model.eval()

    def _prepare_image(self, image):
        if self.feature in ["RGB", "HSV", "LAB"]:
            if self.feature == "HSV":
                return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.feature == "LAB":
                return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            else:
                return image  # RGB
        return image


    def _preprocess_image(self, image):
        image = cv2.resize(image, (256, 256))  # 调整大小
        image = image.transpose((2, 0, 1))  # 转换为 (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).to(self.device) / 255.0
        return image.unsqueeze(0)  # 添加批次维度
    
    def _initialize_mask(self, image_shape):
        return np.zeros(image_shape[:2], np.uint8)


    def _apply_gradient(self, image):
        if self.feature == "sobel":
            image = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
            return image.astype(np.uint8)
        elif self.feature == "prewitt":
            # 这里需要实现 Prewitt 算子
            return prewitt(image)
        elif self.feature == "canny":
            image = cv2.Canny(image, threshold1=100, threshold2=200)
            image = cv2.merge([image, image, image])
            return image
        
        return image

    def _apply_texture(self, image):
        if self.feature == "gabor":
            ksize = 31
            sigma = 4.0
            theta = np.pi / 4
            lamda = 10.0
            gamma = 0.5
            psi = 0
            gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
            return cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
        elif self.feature == "laws":
            # 这里需要实现 Laws 特征提取
            return apply_laws_kernel(image) 
        return image

    def predict(self, images):
        B, W, H, C = images.shape
        output = np.zeros((B, W, H), dtype=np.uint8)

        for i in range(B):
            image = images[i]

            mask = self._initialize_mask(image.shape)

            # 根据 feature 进行处理
            if self.feature in ["RGB", "HSV", "LAB"]:
                image = self._prepare_image(image)
            elif self.feature in ["sobel", "prewitt", "canny"]:
                image = self._apply_gradient(image)

            elif self.feature in ["gabor", "laws"]:
                image = self._apply_texture(image)
            else:
                print("\nFeature", self.feature, "Not Found!")
                return output

            # 使用 grabCut 进行图像分割
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            rect = (10, 10, W-10, H-10)  # 这里的 rect 需要根据实际情况调整
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

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

