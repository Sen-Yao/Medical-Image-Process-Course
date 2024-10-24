import cv2
import numpy as np

def sobel(image, ksize=5):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return gradient_magnitude

def prewitt(image):
    # 如果输入是彩色图像，转换为灰度图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    prewitt_x = np.array([[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1]])

    prewitt_y = np.array([[1, 1, 1],
                           [0, 0, 0],
                           [-1, -1, -1]])

    # 应用 Prewitt 卷积
    gradient_x = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, prewitt_y)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # 归一化到 0-255 范围
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    
    # 将结果堆叠为三通道图像
    gradient_magnitude = gradient_magnitude.astype(np.uint8)
    stacked_image = cv2.merge([gradient_magnitude, gradient_magnitude, gradient_magnitude])

    return stacked_image
