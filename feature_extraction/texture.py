import cv2
import numpy as np

def laws_kernel(kernel_name):
    """生成指定名称的 Laws Kernel"""
    if kernel_name == 'L5':
        return np.array([[1, 4, 6, 4, 1]])
    elif kernel_name == 'E5':
        return np.array([[-1, -2, 0, 2, 1]])
    elif kernel_name == 'S5':
        return np.array([[1, -4, 6, -4, 1]])
    else:
        raise ValueError("Unsupported kernel name")

def create_laws_kernels():
    """生成所有 Laws Kernel 的外积"""
    kernels = []
    for name in ['L5', 'E5', 'S5']:
        kernel = laws_kernel(name)
        for other_name in ['L5', 'E5', 'S5']:
            other_kernel = laws_kernel(other_name)
            combined_kernel = np.outer(kernel.flatten(), other_kernel.flatten())
            kernels.append(combined_kernel)
    return kernels

def apply_laws_kernel(img):
    """应用 Laws Kernel 到输入图像"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laws_kernels = create_laws_kernels()

    # 应用每个 Laws Kernel
    filtered_images = []
    for kernel in laws_kernels:
        filtered_image = cv2.filter2D(img, cv2.CV_64F, kernel)
        
        filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
        
        filtered_images.append(filtered_image)

    # 选择前 3 个过滤后的图像并堆叠
    stacked_images = np.stack(filtered_images[:3], axis=-1).astype('uint8')
    return stacked_images
