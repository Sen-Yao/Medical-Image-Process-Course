import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np


def preprocess_image(img):
    # 将 BGR 转换为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整图像大小
    img = cv2.resize(img, (256, 256))
    
    # 转换为浮点数并归一化
    img = img.astype(np.float32) / 255.0
    
    # 进行标准化
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # 转换为张量并添加批次维度
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # 变为 (1, 3, 224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return img_tensor.to(device)
