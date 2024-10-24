import torch
import cv2
import numpy as np


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (256, 256))
    
    img = img.astype(np.float32) / 255.0
    
    # 进行标准化
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return img_tensor.to(device)
