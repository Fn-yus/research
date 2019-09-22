import cv2
import numpy as np
import glob
import os
import configparser
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from tqdm import tqdm
import csv

def trimming(fname):
    if "Long-needle" in fname:
        img = cv2.imread(fname)
        height = img.shape[0]
        width = img.shape[1]
        img_trimmed = cv2.rotate(img[192:288, 60:565], cv2.ROTATE_180)
        return img_trimmed     
    else:
        pass

def extract_needle(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGRからHSVに変換
    red_lower = np.array([0, 86, 86]) #下限
    red_upper = np.array([30, 255, 255]) #上限

    img_mask = cv2.inRange(hsv, red_lower, red_upper)     #範囲を指定してマスク画像作成
    img_needle = cv2.bitwise_and(img, img, mask=img_mask) #元画像とマスク画像の共通部分を抽出

    return img_needle  

def identify_scale(img, img_needle, fname):
    img_needle_gray = cv2.cvtColor(img_needle, cv2.COLOR_BGR2GRAY)
    img_needle_denoised = cv2.fastNlMeansDenoising(img_needle_gray)
    img_needle_canny = cv2.Canny(img_needle_denoised, 50, 150)
    img_needle_canny2 = cv2.bitwise_not(img_needle_canny)
    ret, img_needle_thresh = cv2.threshold(img_needle_canny2, 127, 255, cv2.THRESH_BINARY)

    needle = 0
    for row in img_needle_thresh:
        needle_list, = np.where(row == 0)
        if len(needle_list) == 2:
            if needle_list[0] >= 460 and np.diff(needle_list) >= 3: #端の場合
                needle = np.mean(needle_list)
                break
            elif np.diff(needle_list) >= 5:
                needle = np.mean(needle_list)
                break
        else:
            continue

    if needle == 0:
        for row in img_needle_thresh:
            needle_list, = np.where(row == 0) #色が黒の箇所を抽出
            print(len(needle_list))
            print(needle_list)
        print(fname)
    else:
        pass

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #グレースケール化
    img_gray_denoised = cv2.fastNlMeansDenoising(img_gray)
    img_canny = cv2.Canny(img_gray_denoised, 70, 150)    
    img_canny2 = cv2.bitwise_not(img_canny)
    ret, img_thresh = cv2.threshold(img_canny2, 127, 255, cv2.THRESH_BINARY)

    img_thresh_diff = cv2.bitwise_not(cv2.subtract(img_needle_thresh, img_thresh))
    if "Long-needle" in fname:
        img_thresh_diff[-1000:1000, 0:20] = 255     #画像左端の映り込み部分を削除
        img_thresh_diff[-1000:1000, 484:505] = 255  #画像右端の映り込み部分を削除

    scale_list = []
    for row in img_thresh_diff:
        black_list, = np.where(row == 0) #色が黒の箇所を抽出
        if len(black_list) == 14 and min(np.diff(black_list[0::2])) > 50: #目盛りの縁が14個かつ目盛り間のピクセル距離が50以上
            scale_list = black_list
            break
        else:
            continue
    
    #うまくいけばここはpass
    if len(scale_list) == 0:
        for row in img_thresh_diff:
            black_list, = np.where(row == 0) #色が黒の箇所を抽出
            print(len(black_list))
            print(black_list)
        print(new_fname)
    else:
        pass

    scale_list_splited = np.split(np.array(scale_list), 7)  #目盛りの左右の線ごとにまとめる

    scales = [np.mean(row) for row in scale_list_splited]   #目盛りの左右の線の平均を取り、scalesにappend

    cv2.line(img, (Decimal(str(needle)).quantize(Decimal("0")), 1000), (Decimal(str(needle)).quantize(Decimal("0")), -1000), (0, 0, 255), 1)
    for i in scales:
        cv2.line(img, (Decimal(str(i)).quantize(Decimal("0")), 1000), (Decimal(str(i)).quantize(Decimal("0")), -1000), (0, 255, 0), 1)

    cv2.imwrite('results/pictures/img.jpg', img)
    cv2.imwrite('results/pictures/img_needle_thresh.jpg', img_needle_thresh)
    cv2.imwrite('results/pictures/img_thresh.jpg', img_thresh)
    cv2.imwrite('results/pictures/img_thresh_diff.jpg', img_thresh_diff)

    return needle, scales

def digitalize(needle, scales):
    scale_upper_list, = np.where(needle <= scales)
    scale_lower_list, = np.where(needle >= scales)

    if len(scale_upper_list) == 0: #一番右の目盛りより右側に針がある場合
        scales_diff = np.diff(scales, n = 2)
        scale_upper = (2 * scales_diff[4] - scales_diff[3]) + np.diff(scales, n = 1)[5] + scales[6] #目盛り4と3の差分計算（値はマイナス）し、それをそれぞれに足していく

        scale_lower_index = max(scale_lower_list)
        scale_lower = scales[scale_lower_index]

        needle_percentage = (needle - scale_lower)/(scale_upper - scale_lower)
        needle_position = (scale_lower_index - 3) + needle_percentage

        return needle_position

    elif len(scale_lower_list) == 0: #一番左の目盛りより左側に針がある場合
        scale_upper = scales[min(scale_upper_list)]

        scale_lower_index = -1
        scales_diff = np.diff(scales, n = 2)
        scale_lower = (2 * scales_diff[0] - scales_diff[1]) - np.diff(scales, n = 1)[0] + scales[0]

        needle_percentage = (needle - scale_lower)/(scale_upper - scale_lower)
        needle_position = (scale_lower_index - 3) + needle_percentage

        return needle_position
    
    else:
        scale_upper = scales[min(scale_upper_list)]

        scale_lower_index = max(scale_lower_list)
        scale_lower = scales[scale_lower_index]

        needle_percentage = (needle - scale_lower)/(scale_upper - scale_lower)
        needle_position = (scale_lower_index - 3) + needle_percentage

        return needle_position

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config/config.ini")
    path = config.get('path', 'long_needle')
    files = glob.glob(path) 
    
    for fname in tqdm(files):
        new_fname, ext = os.path.splitext(os.path.basename(fname))
        img = trimming(fname)
        img_needle = extract_needle(img)
        identifyscale = identify_scale(img, img_needle, fname)
        #print("=============================================")
        #print(new_fname)
        #print("\n")
        #print("・針の位置：" + str(identifyscale[0]))
        #print("・目盛り座標：" + str(identifyscale[1]))
        #print("・目盛り幅：" + str(np.diff(identifyscale[1], n = 1)))
        #print("・針の座標：" + str(digitalize(identifyscale[0], identifyscale[1])))
        #print("=============================================")
