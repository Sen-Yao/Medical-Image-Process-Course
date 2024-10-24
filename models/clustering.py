import numpy as np
import cv2
import torch

from sklearn.cluster import KMeans

from feature_extraction.intensity import sobel, prewitt
from feature_extraction.texture import apply_laws_kernel
from feature_extraction.ResNet import preprocess_image

from torchvision import models

class ColorSpace_Clustering:
    def __init__(self, k=2, color_space="RGB"):
        self.k = k
        self.color_space = color_space
    def predict(self, images):
        # 确保输入是一个批量图像
        if len(images) == 0 or images[0].shape[2] != 3:
            raise ValueError("Input must be a list of RGB images with shape (W, H, C)")

        binary_masks = []

        for image in images:
            if self.color_space == "HSV":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.color_space == "LAB":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            pixel_values = image.reshape((-1, 3))
            pixel_values = pixel_values.astype(np.float32) / 255.0
            
            kmeans = KMeans(n_clusters=self.k, random_state=0)
            kmeans.fit(pixel_values)

            labels = kmeans.labels_
            binary_mask = labels.reshape(image.shape[:2])

            binary_mask = (binary_mask == 0).astype(np.uint8)
            binary_masks.append(binary_mask)

        return np.array(binary_masks)
    

class ColorSpace_Grad_Clustering:
    def __init__(self, k=2, color_space="RGB", grad=None):
        self.k = k
        self.color_space = color_space
        self.grad = grad
    
    def predict(self, images):
        # 确保输入是一个批量图像
        if len(images) == 0 or images[0].shape[2] != 3:
            raise ValueError("Input must be a list of RGB images with shape (W, H, C)")

        binary_masks = []

        for image in images:
            if self.color_space == "HSV":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.color_space == "LAB":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            pixel_values = image.reshape((-1, 3))
            pixel_values = pixel_values.astype(np.float32) / 255.0

            if self.grad == "sobel":
                gradient_magnitude = sobel(image).reshape((-1, 3)) 
            elif self.grad == "prewitt":
                gradient_magnitude = prewitt(image).reshape((-1, 3)) 
            elif self.grad == "canny":
                gradient_magnitude = cv2.Canny(image, threshold1=100, threshold2=200).reshape((-1, 1)) 
            else:
                gradient_magnitude = np.zeros(pixel_values.shape[0])
                gradient_magnitude = gradient_magnitude.reshape((-1, 1)) 

            np.concatenate((pixel_values, gradient_magnitude), axis=1)
            
            kmeans = KMeans(n_clusters=self.k, random_state=0)
            kmeans.fit(gradient_magnitude)

            labels = kmeans.labels_
            binary_mask = labels.reshape(image.shape[:2])

            binary_mask = (binary_mask == 0).astype(np.uint8)
            binary_masks.append(binary_mask)

        return np.array(binary_masks) 
    

class ColorSpace_Texture_Clustering:
    def __init__(self, k=2, color_space="RGB", texture=None):
        self.k = k
        self.color_space = color_space
        self.texture = texture
    def predict(self, images):
        # 确保输入是一个批量图像
        if len(images) == 0 or images[0].shape[2] != 3:
            raise ValueError("Input must be a list of RGB images with shape (W, H, C)")

        binary_masks = []

        for image in images:
            if self.color_space == "HSV":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.color_space == "LAB":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            pixel_values = image.reshape((-1, 3))
            pixel_values = pixel_values.astype(np.float32) / 255.0
            
            if self.texture == "gabor":
                # 设置 Gabor 滤波器的参数
                ksize = 31  # 滤波器的大小
                sigma = 4.0  # 高斯函数的标准差
                theta = np.pi / 4  # 滤波器的方向
                lamda = 10.0  # 波长
                gamma = 0.5  # 纵横比
                psi = 0  # 相位偏移

                gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)

                feature_img = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel).reshape((-1, 3)) 

            elif self.texture == "laws":
                feature_img = apply_laws_kernel(image).reshape((-1, 3)) 
            else:
                feature_img = np.zeros(pixel_values.shape[0]).reshape((-1, 1)) 

            np.concatenate((pixel_values, feature_img), axis=1)

            kmeans = KMeans(n_clusters=self.k, random_state=0)
            kmeans.fit(feature_img)

            labels = kmeans.labels_
            binary_mask = labels.reshape(image.shape[:2])

            binary_mask = (binary_mask == 0).astype(np.uint8)
            binary_masks.append(binary_mask)

        return np.array(binary_masks) 


class ResNet_Clustering:
    def __init__(self, k=2):
        self.k = k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT').to(self.device)
        self.model.eval()
    def preprocess_image(self, image):
        image = cv2.resize(image, (256, 256))
        image = image.transpose((2, 0, 1))  # 转换为 (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).to(self.device) / 255.0
        return image.unsqueeze(0)  # 添加批次维度

    def predict(self, images):
        # 确保输入是一个批量图像
        if len(images) == 0 or images[0].shape[2] != 3:
            raise ValueError("Input must be a list of RGB images with shape (W, H, C)")
        
        # 预处理所有图像并将它们堆叠成一个批次
        img_tensors = torch.cat([self.preprocess_image(image) for image in images], dim=0)

        with torch.no_grad():
            features = self.model(img_tensors)['out']
        
        features = features.cpu().numpy()
        N, C, H, W = features.shape
        features = features.reshape((N, H * W, C))

        binary_masks = []

        for i in range(len(images)):
            kmeans = KMeans(n_clusters=self.k, random_state=0)
            kmeans.fit(features[i])

            labels = kmeans.labels_
            binary_mask = labels.reshape((images[i].shape[:2]))

            binary_mask = (binary_mask == 0).astype(np.uint8)
            binary_masks.append(binary_mask)

        return np.array(binary_masks)