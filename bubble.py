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
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #グレースケール化
    img_gray_denoised = cv2.fastNlMeansDenoising(img_gray)
    img_canny = cv2.Canny(img_gray_denoised, 280, 330)
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

    cv2.imshow('img_thresh.jpg', img_thresh)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config/config.ini")
    target_path = config.get('path', 'long-bubble')
    target_files = glob.glob(target_path)
    fname = target_files[0] #決め打ち
    img = trimming(fname)
    identify_scale(img, fname)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

