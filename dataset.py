import glob
import os
import torch
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader

class MedicalDataset(Dataset):
    def __init__(self, data_name=None, crop_size=384):
        self.data_name = data_name
        data_dirs = "dataset/"+ data_name
        if data_dirs is None:
            data_dirs = ['dataset/Data']
        elif isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.crop_size = crop_size
        print(data_dirs)
        self.data_paths = []

        for data_dir in data_dirs:
            if data_dir == 'dataset/Data':
                trainset_name = 'Image'
                annotation_name = 'Annotation'
            elif data_dir == 'dataset/GlaS':
                trainset_name = 'train'
                annotation_name = 'trainanno'
            elif data_dir == 'dataset/UDIAT':
                trainset_name = 'imgs'
                annotation_name = 'gt' 
            else:
                print("Unknow Dataset:", data_dir)
                break

            image_dirs = sorted(glob.glob(os.path.join(data_dir, trainset_name, '*')))
            
            # 确保图像和注释一一对应
            for image_path in image_dirs:
                image_name = os.path.basename(image_path)
                annotation_path = os.path.join(data_dir, annotation_name, image_name)
                
                if os.path.exists(annotation_path):
                    self.data_paths.append((image_path, annotation_path))
                
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_path, annotation_path = self.data_paths[idx]
        
        # 使用 OpenCV 加载图像和注释
        image = cv2.imread(image_path)
        annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)  # 假设注释是单通道图像
        
        # 检查图像是否成功加载
        if image is None or annotation is None:
            raise FileNotFoundError(f"Image or annotation not found: {image_path} or {annotation_path}")

        # 进行裁剪或其他预处理
        if self.crop_size:
            image = cv2.resize(image, (self.crop_size, self.crop_size))
            annotation = cv2.resize(annotation, (self.crop_size, self.crop_size))

        # 将图像从 BGR 转换为 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, annotation