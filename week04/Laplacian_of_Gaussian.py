import numpy as np
import cv2
import os
import time

def my_padding(src, filter):
    (h, w) = src.shape
    (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h+h_pad*2, w+w_pad*2))
    padding_img[h_pad:h+h_pad, w_pad:w+w_pad] = src
    return padding_img

def my_filtering(src, filter):
    (h, w) = src.shape
    (m_h, m_w) = filter.shape
    pad_img = my_padding(src, filter)
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * filter)
    return dst

def gaussian_2D_filtering(filter_size, sigma):

    y, x = np.mgrid[-(filter_size // 2) : (filter_size // 2) + 1, -(filter_size // 2) : (filter_size // 2) + 1]

    scalar = 1 / (2 * np.pi * sigma ** 2)
    exponential = np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    mask = scalar * exponential
    gaussian_mask = mask / np.sum(mask)

    return gaussian_mask

def calculate_magnitude(sobel_x, sobel_y):

    # Element Wise Multiplication

    IxIx = np.zeros(sobel_x.shape, dtype=sobel_x.dtype)
    for i in range(IxIx.shape[0]):
        for j in range(IxIx.shape[1]):
            IxIx[i,j] = sobel_x[i,j] * sobel_x[i,j]

    IyIy = np.zeros(sobel_y.shape, dtype=sobel_x.dtype)
    for i in range(IyIy.shape[0]):
        for j in range(IyIy.shape[1]):
            IyIy[i, j] = sobel_y[i, j] * sobel_y[i, j]

    magnitude = np.sqrt(IxIx + IyIy)

    # Simple
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return magnitude

def generate_laplacian_filter():
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    return laplacian

def get_LoG_filter(fsize, sigma=1):
    y, x = np.mgrid[-(fsize // 2):(fsize // 2) + 1, -(fsize // 2):(fsize // 2) + 1]

    # 수식 구현
    LoG_x = (((x ** 2) / sigma ** 4) - (1 / (sigma ** 2))) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    LoG_y = (((y ** 2) / sigma ** 4) - (1 / (sigma ** 2))) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    LoG = LoG_x + LoG_y

    # 필터의 총 합을 0으로 만들기
    print(LoG)

    return LoG

if __name__ == "__main__":
    src_path = 'Lena.png'
    src_image = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

    # Laplacian of Gaussian
    LoG = get_LoG_filter(11, 1)
    dst = my_filtering(src_image, LoG)

    cv2.imshow('Original', src_image)
    cv2.imshow('LoG', dst/np.max(dst))
    cv2.waitKey()
    cv2.destoryAllWindows()

    gaus = gaussian_2D_filtering(11, 1)
    sobel = generate_laplacian_filter()
    gaus_filtering = my_filtering(src_image, gaus)
    dst = my_filtering(gaus_filtering, sobel)

    cv2.imshow('LoG', dst / np.max(dst))
    cv2.waitKey()
    cv2.destoryAllWindows()
