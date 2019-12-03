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
        img_trimmed = cv2.rotate(img[90:430, 235:390], cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_trimmed

def identify_scale(img, fname):   
    cv2.imwrite('results/pictures/bubble/img.jpg', img)

    gamma = 0.1
    lookuptable = np.zeros((256,1),dtype = 'uint8')
    for i in range(256):
        lookuptable[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)
    
    img_darked = cv2.LUT(img, lookuptable)

    #img_scale_mask = cv2.inRange(img_darked, (0, 0, 0), (40, 40, 40))  #黒色
    #img_scale_masked = cv2.bitwise_and(img, img, mask=img_scale_mask) #元画像とマスク画像の共通部分を抽出

    cv2.imwrite('results/pictures/bubble/img_darked.jpg', img_darked)

    img_gray = cv2.cvtColor(img_darked, cv2.COLOR_BGR2GRAY) #グレースケール化
    img_gray_denoised = cv2.fastNlMeansDenoising(img_gray)
    #img_canny = cv2.Canny(img_gray_denoised, 300, 350)  #要調整
    #img_canny2 = cv2.bitwise_not(img_canny)
    ret, img_thresh = cv2.threshold(img_gray_denoised, 15, 255, cv2.THRESH_BINARY)
    img_thresh_canny = cv2.Canny(img_thresh, 0, 100)
    img_thresh_canny2 = cv2.bitwise_not(img_thresh_canny)

    cv2.imwrite('results/pictures/bubble/img_gray.jpg', img_gray_denoised)
    cv2.imwrite('results/pictures/bubble/img_thresh.jpg', img_thresh)
    cv2.imwrite('results/pictures/bubble/img_thresh_canny.jpg', img_thresh_canny)    

    scale_list = []
    for row in img_thresh_canny2:
        black_list, = np.where(row == 0) #色が黒の箇所を抽出
        if "bubble" in fname.lower():
            if len(black_list) == 16: #目盛りの縁が16個
                if min(np.diff(black_list[0::2])) > 20: #目盛り間の幅が20ピクセル以上
                    if black_list[8] - black_list[7] > 150: #左側と右側の目盛りのギャップを確実に
                        scale_list.append(black_list)
                        # print(black_list)

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

    # print(scales)

    for i in scales:
        cv2.line(img, (Decimal(str(i)).quantize(Decimal("0")), 1000), (Decimal(str(i)).quantize(Decimal("0")), -1000), (0, 255, 0), 1)
    
    cv2.imwrite('results/pictures/bubble/img_thresh_canny2.jpg', img_thresh_canny2)
    cv2.imwrite('results/pictures/bubble/img_lined.jpg'.format(fname), img)

    return scales, img_gray_denoised

def identify_bubble(fname, img_origin, img):
    # ret, img_bubble_thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    # img_bubble_thresh_canny = cv2.Canny(img_bubble_thresh, 0, 100)
    img_bubble_canny = cv2.Canny(img, 250, 550)
    # cv2.imwrite('results/pictures/bubble/img_bubble_thresh.jpg', img_bubble_thresh)
    # cv2.imwrite('results/pictures/bubble/img_bubble_thresh_canny.jpg', img_bubble_thresh_canny)
    cv2.imwrite('results/pictures/bubble/img_bubble_canny.jpg', img_bubble_canny)

    contours, _ = cv2.findContours(img_bubble_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    # with open('results/data/bubble/contours.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(contours)
    cnt_list = None
    for cnt in contours:
        if len(cnt) >= 5: #cv2.fitEllipseは、最低5つの点がないとエラーを起こすため
            (x, y), (long_rad, short_rad), angle = cv2.fitEllipse(cnt) #(x,y)は楕円の中心の座標、(MA, ma)はそれぞれ長径,短径、angleは楕円の向き(0≤angle≤180, 0が鉛直方向)
            # print(ellipse)
            if 80 < angle < 100 and long_rad >= 10 and short_rad >= 10: #楕円の向きを絞り,直線を近似しているものは排除する
                # print([x, y, long_rad, short_rad])
                # print(cnt[:, 0, 0]) 
                cnt_left_edge  = min(cnt[:,0,0]) #3次元numpy配列になっている（[[[a, b]], [[c, d]]]という形）
                cnt_right_edge = max(cnt[:,0,0])
                cnt_upper_edge = min(cnt[:,0,1])
                cnt_lower_edge = max(cnt[:,0,1])
                x_cal          = (cnt_left_edge  + cnt_right_edge) * 0.5
                y_cal          = (cnt_upper_edge + cnt_lower_edge) * 0.5
                long_rad_cal   = cnt_right_edge - x_cal
                short_rad_cal  = cnt_lower_edge - y_cal
                cnt_RMS        = ((x_cal - x) ** 2 + (y_cal - y) ** 2 + (long_rad_cal - long_rad) ** 2 + (short_rad_cal - short_rad) ** 2) ** 0.5

                if cnt_list == None or cnt_RMS < cnt_list[1]:
                    cnt_list = [cnt, cnt_RMS]
                else:
                    continue

    target_cnt = cnt_list[0][:,0] #3次元配列を2次元配列に（[[[a, b]], [[c, d]]] => [[a, b], [c,d]]）
    # print(cnt_list)
    # print(target_cnt)
    img_ellipse = cv2.drawContours(img_origin, [cnt_list[0]], 0, (0, 0, 255), 2)
    cv2.imwrite('results/pictures/bubble/img_ellipse.jpg', img_ellipse)

    return target_cnt

def cross_correlation(fname, img_origin, img, origin_cnt):
    img_bubble_canny = cv2.Canny(img, 250, 550)
    contours, _      = cv2.findContours(img_bubble_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 以下のループ処理はかなり重いと予想されるため、findContoursの第3引数をcv2.CHAIN_APPBOX_SIMPLEにすることも考える(精度に関しては要検証)
    # 2019/12/03 研究室PC, cv2.CHAIN_APPBOX_SIMPLE, (x, y) = (-100~100, -1~1) で 69分予想（開始11分段階） -> 結果1時間ほどかかった

    correlation_list = None
    for cnt in contours:
        for x in range(-100, 100):
            for y in range(-10, 10):
                # 全ての行が[x, y]のcontoursのサイズを持つnumpy配列を作成し足す
                origin_cnt_size = len(origin_cnt)
                cross_list      = np.array([[x,y] for i in range(origin_cnt_size)])
                cross_cnt       = origin_cnt + cross_list
                cross_count     = 0
                for cross_pixel in cross_cnt:
                    cross_pixel_count = np.count_nonzero((cnt[:, 0] == cross_pixel).all(axis=1)) #行方向に対して配列ごとに一致しているかどうかをBooleanで判断し、Trueの数を数える
                    # print((cnt[:,0] == cross_pixel).all(axis=1))
                    if cross_pixel_count != 0: #速くなったりしないかな
                        cross_count += cross_pixel_count
                if correlation_list == None or cross_count > correlation_list[3]:
                    correlation_list = [cnt, x, y, cross_count]
    
    img_contour = cv2.drawContours(img_origin, [correlation_list[0]], 0, (0, 0, 255), 2)

    created_unix_time = os.path.getmtime(fname)
    created_datetime  = datetime.fromtimestamp(created_unix_time)
    dt = created_datetime.strftime('%Y%m%d%H%M%S')
    cv2.imwrite('results/pictures/contour/img_contour_{}.jpg'.format(dt), img_contour)
    cv2.imwrite('results/pictures/canny/img_canny_{}.jpg'.format(dt), img_bubble_canny)

    with open('results/data/bubble/corr.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        csv_list =sum([[dt], correlation_list[1:]], [])
        writer.writerow(csv_list)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config/config.ini")
    target_path = config.get('path', 'long-bubble')
    target_files = glob.glob(target_path)
    csv_lists = []
    target_cnt = None
    for fname in tqdm(target_files): #決め打ち
        # print(fname)
        img = trimming(fname)
        identify_scale_list = identify_scale(img, fname)
        scales = identify_scale_list[0]
        if fname == target_files[0]:
            target_cnt = identify_bubble(fname, img, identify_scale_list[1])
            cross_correlation(fname, img, identify_scale_list[1], target_cnt)
        else:
            cross_correlation(fname, img, identify_scale_list[1], target_cnt)
        csv_list = sum([[fname], scales], []) #平坦化している
        csv_lists.append(csv_list)

    with open('results/data/bubble/a.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_lists)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

