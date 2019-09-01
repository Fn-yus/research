import cv2
import numpy as np

img = cv2.imread("pictures/sample_1.jpg") #画像読み込み

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGRからHSVに変換
red_lower = np.array([0, 86, 86]) #下限
red_upper = np.array([30, 255, 255]) #上限

img_mask = cv2.inRange(hsv, red_lower, red_upper) #範囲を指定してマスク画像作成
img_needle = cv2.bitwise_and(img, img, mask=img_mask) #元画像とマスク画像の共通部分を抽出

cv2.imwrite("pictures/sample_1_needle.jpg", img_needle) #画像書き出し
    



