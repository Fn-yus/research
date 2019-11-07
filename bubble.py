import cv2
import numpy as np
import matplotlib.pyplot as plt
from decimal import *
from datetime import datetime, timedelta
from time import sleep
import glob
import os
import configparser
import csv
from tqdm import tqdm

def trimming(fname):
    img = cv2.imread(fname)
    img_trimmed = None
    if "long-bubble" in fname.lower():
        img_trimmed = cv2.rotate(img[80:440, 235:390], cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_trimmed

def identify_scale(img, fname):   
    cv2.imwrite('img.jpg', img)

    gamma = 0.1
    lookuptable = np.zeros((256,1),dtype = 'uint8')
    for i in range(256):
        lookuptable[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)
    
    img_darked = cv2.LUT(img, lookuptable)

    img_scale_mask = cv2.inRange(img_darked, (0, 0, 0), (40, 40, 40))  #黒色
    img_scale_masked = cv2.bitwise_and(img, img, mask=img_scale_mask) #元画像とマスク画像の共通部分を抽出

    cv2.imwrite('img_darked.jpg', img_darked)
    cv2.imwrite('img.jpg', img_scale_mask)

    img_gray = cv2.cvtColor(img_scale_masked[60:90, 10:350], cv2.COLOR_BGR2GRAY) #グレースケール化
    img_gray_denoised = cv2.fastNlMeansDenoising(img_gray)
    img_canny = cv2.Canny(img_gray_denoised, 50, 100)  #要調整
    img_canny2 = cv2.bitwise_not(img_canny)
    ret, img_thresh = cv2.threshold(img_canny2, 127, 255, cv2.THRESH_BINARY)

    scale_list = []
    for row in img_thresh:
        black_list, = np.where(row == 0) #色が黒の箇所を抽出
        if "bubble" in fname.lower():
            if len(black_list) == 16: #目盛りの縁が16個
                scale_list.append(black_list)

    if len(scale_list) == 0:
        for row in img_thresh:
            black_list, = np.where(row == 0) #色が黒の箇所を抽出
            print(len(black_list))
            print(black_list)
        print(fname)
    else:
        pass

    scale_list = np.mean(np.array(scale_list), axis=0)      #条件に一致する線の平均をリストに
    scale_list_splited = np.split(np.array(scale_list), 8)  #目盛りの左右の線ごとにまとめる
    scales = [np.mean(row) for row in scale_list_splited]   #目盛りの左右の線の平均を取り、scalesにappend

    for i in scales:
        cv2.line(img_thresh, (Decimal(str(i)).quantize(Decimal("0")), 1000), (Decimal(str(i)).quantize(Decimal("0")), -1000), (0, 255, 0), 1)

    cv2.imwrite('img_gray.jpg', img_gray)
    cv2.imwrite('img_gray_denoised.jpg', img_gray_denoised)
    cv2.imwrite('img_canny.jpg', img_canny)
    cv2.imwrite('img_thresh.jpg'.format(fname), img_thresh)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config/config.ini")
    target_path = config.get('path', 'long-bubble')
    target_files = glob.glob(target_path)
    for fname in target_files: #決め打ち
        sleep(1)
        print(fname)
        img = trimming(fname)
        identify_scale(img, fname)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

