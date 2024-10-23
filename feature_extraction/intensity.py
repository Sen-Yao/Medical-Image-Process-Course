import cv2
import numpy as np

def sobel(image, ksize=5):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return gradient_magnitude

def prewitt(image):
    prewitt_x = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])

    prewitt_y = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]])

    # 应用 Prewitt 卷积
    gradient_x = cv2.filter2D(image, -1, prewitt_x)
    gradient_y = cv2.filter2D(image, -1, prewitt_y)

    # 计算梯度幅度
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return gradient_magnitude.astype(np.uint8)