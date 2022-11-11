import numpy as np
import cv2

def find_local_maxima(src, ksize):
    (h, w) = src.shape
    pad_img = np.zeros((h+ksize, w+ksize))
    pad_img[ksize//2:h+ksize//2, ksize//2:w+ksize//2] = src
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            max_val = np.max(pad_img[row : row+ksize, col:col+ksize])
            if max_val == 0:
                continue
            if src[row, col] == max_val:
                dst[row, col] = src[row, col]
    return dst

  
def main():
    """
    Harris corner detector.
    cv2.cornerHarris(gray, 3, 3, 0.04)
    :parameter
    """
    src = cv2.imread('zebra.png')
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY).astype(np.float32)

    # harris corner 라이브러리 사용
    dst = cv2.cornerHarris(gray, 3, 3, 0.04)

    # 코너점들이 너무 작아서 점들을 시각적으로 잘 보이기 위해서 하는 작업
    dst = cv2.dilate(dst, None)
    cv2.imshow('original',src)
    har = ((dst - np.min(dst)) / np.max(dst - np.min(dst)) * 255 + 0.5).astype(np.uint8)
    cv2.imshow('har',har)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # thresholding
    dst[dst < 0.01 * dst.max()] = 0
    # non-max suppresion을 진행하여 코너점들이 많이 몰려 있는 것을 극댓값만 남기고 나머지는 버림
    dst = find_local_maxima(dst,21)
    cv2.imshow('harris_dst', dst)

    # 이미지의 코너마다 빨간점 찍기
    #dst.shape[0] : dst의 행
    #dst.shape[1] : dst의 열
    #3 : 채널
    interest_points = np.zeros((dst.shape[0],dst.shape[1],3))
    
    # 빨간점
    # [B, G ,R ] == [0, 0, 255]
    interest_points[dst != 0] =[0,0,255]
    cv2.imshow('interest_points',interest_points)

    #원본 이미지에 코너마다 빨간점(코너) 찍기
    src[dst != 0] = [0,0,255]

    cv2.imshow('harris',src)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
