import numpy as np
import cv2

def check_unique_labels(image):
    """
    检查图像中的唯一标签。

    参数:
    image (np.ndarray): 输入图像，假设为二维或三维 NumPy 数组。

    返回:
    unique_labels (np.ndarray): 图像中的唯一标签。
    """
    # 确保输入是 NumPy 数组
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # np.set_printoptions(threshold=np.inf)
    print(image)

    # 将图像扁平化并获取唯一标签
    unique_labels = np.unique(image)

    return unique_labels

