import cv2
import numpy as np
from decimal import Decimal
from datetime import datetime
from time import sleep
import glob
import os
import configparser
import csv
from tqdm import tqdm
from itertools import product
import plot

def trimming(fname):
    img = cv2.imread(fname)
    img_trimmed = None
    if "long-bubble" in fname.lower():
        img_trimmed = cv2.rotate(img[90:430, 235:390], cv2.ROTATE_90_COUNTERCLOCKWISE)
    if "cross-bubble" in fname.lower():
        img_trimmed = cv2.rotate(img[259:365, 120:515], cv2.ROTATE_180)
    return img_trimmed

def identify_scale(img, fname):   
    cv2.imwrite('../results/pictures/bubble/img.jpg', img)

    gamma = 0.1
    lookuptable = np.zeros((256,1), dtype='uint8')
    for i in range(256):
        lookuptable[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)
    
    img_darked = cv2.LUT(img, lookuptable)

    #img_scale_mask = cv2.inRange(img_darked, (0, 0, 0), (40, 40, 40))  #黒色
    #img_scale_masked = cv2.bitwise_and(img, img, mask=img_scale_mask) #元画像とマスク画像の共通部分を抽出

    cv2.imwrite('../results/pictures/bubble/img_darked.jpg', img_darked)

    img_gray = cv2.cvtColor(img_darked, cv2.COLOR_BGR2GRAY) #グレースケール化
    img_gray_denoised = cv2.fastNlMeansDenoising(img_gray)
    #img_canny = cv2.Canny(img_gray_denoised, 300, 350)  #要調整
    #img_canny2 = cv2.bitwise_not(img_canny)
    ret, img_thresh = cv2.threshold(img_gray_denoised, 15, 255, cv2.THRESH_BINARY)
    img_thresh_canny = cv2.Canny(img_thresh, 0, 100)
    img_thresh_canny2 = cv2.bitwise_not(img_thresh_canny)

    cv2.imwrite('../results/pictures/bubble/img_gray.jpg', img_gray_denoised)
    cv2.imwrite('../results/pictures/bubble/img_thresh.jpg', img_thresh)
    cv2.imwrite('../results/pictures/bubble/img_thresh_canny.jpg', img_thresh_canny)    

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
    
    cv2.imwrite('../results/pictures/bubble/img_thresh_canny2.jpg', img_thresh_canny2)
    cv2.imwrite('../results/pictures/bubble/img_lined.jpg'.format(fname), img)

    return scales, img_gray_denoised

def identify_bubble(fname, img_origin, img):
    # ret, img_bubble_thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    # img_bubble_thresh_canny = cv2.Canny(img_bubble_thresh, 0, 100)
    img_bubble_canny = None
    if "long-bubble" in fname.lower():
        img_bubble_canny = cv2.Canny(img, 250, 550)
    elif "cross-bubble" in fname.lower():
        img_bubble_canny = cv2.Canny(img, 250, 550)
    # cv2.imwrite('results/pictures/bubble/img_bubble_thresh.jpg', img_bubble_thresh)
    # cv2.imwrite('results/pictures/bubble/img_bubble_thresh_canny.jpg', img_bubble_thresh_canny)
    # cv2.imwrite('results/pictures/bubble/img_gray_denoised.jpg', img)
    # cv2.imwrite('results/pictures/bubble/img_bubble_canny.jpg', img_bubble_canny)

    contours, _ = cv2.findContours(img_bubble_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    # with open('results/data/bubble/contours.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(contours)
    contours_for_loop = None

    if "long-bubble" in fname.lower():
        contours_for_loop = contours
    elif "cross-bubble" in fname.lower():
        ndarray_contours = np.array(contours)
        filtered_contours = np.concatenate([contours[4], contours[8]])
        contours_for_loop = [filtered_contours]

    cnt_list = None        

    for cnt in contours_for_loop:
        if len(cnt) >= 5: #cv2.fitEllipseは、最低5つの点がないとエラーを起こすため
            (x, y), (long_rad, short_rad), angle = cv2.fitEllipse(cnt) #(x,y)は楕円の中心の座標、(MA, ma)はそれぞれ長径,短径、angleは楕円の向き(0≤angle≤180, 0が鉛直方向)
            # img_all_contour = cv2.ellipse(img_origin, ((x, y), (long_rad, short_rad), angle), (0,255,0), 2)
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
                if cnt_list is None or cnt_RMS < cnt_list[1]:
                    cnt_list = [cnt, x_cal, cnt_RMS]
                else:
                    continue
                # elif "cross-bubble" in fname.lower():
                #     if cnt_first_list == None:
                #         cnt_first_list = [cnt, x_cal, cnt_RMS]
                #     elif cnt_second_list == None:
                #         if cnt_RMS < cnt_first_list[1]:
                #             cnt_second_list = cnt_first_list
                #             cnt_first_list = [cnt, x_cal, cnt_RMS]
                #         else:
                #             cnt_second_list = [cnt, x_cal, cnt_RMS]
                #     elif cnt_RMS < cnt_first_list[1]:
                #         cnt_second_list = cnt_first_list
                #         cnt_first_list = [cnt, x_cal, cnt_RMS]
                #     elif cnt_RMS < cnt_second_list[1]:
                #         cnt_second_list = [cnt, x_cal, cnt_RMS]
                #     else:
                #         continue

#    if "cross-bubble" in fname.lower():
#        with open('a.csv', 'w', newline='') as f:
#            writer = csv.writer(f)
#            writer.writerows(contours)

    target_cnt = cnt_list[0][:,0] #3次元配列を2次元配列に（[[[a, b]], [[c, d]]] => [[a, b], [c,d]]）
    img_ellipse = cv2.drawContours(img_origin, [cnt_list[0]], 0, (255, 0, 0), 1)
        
    # print(cnt_list)
    # print(target_cnt)
    # cv2.imwrite('results/pictures/bubble/img_all_contour.jpg', img_all_contour)

    cv2.imwrite('../results/pictures/bubble/img_ellipse.jpg', img_ellipse)

    return target_cnt, cnt_list[1]

def cross_correlation(fname, img_origin, img, origin_cnt, created_datetime_second):
    img_bubble_canny = None
    if "long-bubble" in fname.lower():
        img_bubble_canny = cv2.Canny(img, 300, 750)
    elif "cross-bubble" in fname.lower():
        img_bubble_canny = cv2.Canny(img, 200, 750)
    contours, _      = cv2.findContours(img_bubble_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 以下のループ処理はかなり重いと予想されるため、findContoursの第3引数をcv2.CHAIN_APPBOX_SIMPLEにすることも考える(精度に関しては要検証)
    # 2019/12/03 研究室PC, cv2.CHAIN_APPBOX_SIMPLE, (x, y) = (-100~100, -1~1)   で 69分予想（開始11分段階） -> 結果1時間ほどかかった
    # 2019/12/03 研究室PC, cv2.CHAIN_APPROX_NONE,   (x, y) = (-100~100, -10~10) で 12時間かかった
    # 2019/12/03 研究室PC, cv2.CHAIN_APPROX_NONE,   (x, y) = (-75~75,   0) で 25分
    # 2019/12/03 研究室PC, cv2.CHAIN_APPROX_SIMPLE  (x, y) = (-75~75,   0) で 18分

    correlation_list = None
    if created_datetime_second >= 10:
        for cnt, x in product(contours, range(-60, 60)):
            # for y in range(-10, 10): # y = [-10, 10] で実行しても解の幅は -1 ~ 1 だったのでとりあえずなくす
            # 全ての行が[x, y]のcontoursのサイズを持つnumpy配列を作成し足す
            origin_cnt_size = len(origin_cnt)
            cross_list      = np.array([[x,0] for i in range(origin_cnt_size)])
            cross_cnt       = origin_cnt + cross_list
            cross_count     = 0
            for cross_pixel in cross_cnt:
                cross_pixel_count = np.count_nonzero((cnt[:, 0] == cross_pixel).all(axis=1)) #行方向に対して配列ごとに一致しているかどうかをBooleanで判断し、Trueの数を数える
                # print((cnt[:,0] == cross_pixel).all(axis=1))
                cross_count += cross_pixel_count
            if correlation_list is None or cross_count > correlation_list[3]:
                correlation_list = [cnt, cross_cnt, x, cross_count]

        datetime_number, _ = os.path.splitext(os.path.basename(fname))

        img_contour = cv2.drawContours(img_origin, [correlation_list[0]], 0, (0, 0, 255), 2)
        cv2.imwrite('../results/pictures/contour/img_contour_{}.jpg'.format(datetime_number), img_contour)

        img_wrapped = cv2.drawContours(img_contour, [correlation_list[1]], 0, (255, 0, 0), 1)

        cv2.imwrite('../results/pictures/canny/img_canny_{}.jpg'.format(datetime_number), img_bubble_canny)
        cv2.imwrite('../results/pictures/wrapped/img_wrapped_{}.jpg'.format(datetime_number), img_wrapped)

        corr_list = correlation_list[2:] 
        return corr_list
    else:
        corr_list = None
        return corr_list

def bubble_position(fname, scales, corr_list, x_origin):
    origin_line = (scales[3] + scales[4]) * 0.5
    scales_left = np.array(scales[0:4])
    scales_right = np.array(scales[4:8])
    scales_left_width_ave_list = np.diff(scales_left)
    scales_right_width_ave_list = np.diff(scales_right)
    scale_width_ave = np.mean(np.append(scales_left_width_ave_list, scales_right_width_ave_list))

    bubble_pixel = x_origin + corr_list[0]
    bubble_position = (bubble_pixel - origin_line) / scale_width_ave
    return bubble_position

if __name__ == "__main__":
    target = None
    while True:
        target = input("解析する画像の種類を選んでください\n[1.long-bubble, 2.cross-bubble]：")
        if target.lower() == "long-bubble" or target.lower() == "cross-bubble":
            sleep(0.3)
            print("ok\n")
            break
        elif target == "1":
            sleep(0.3)
            target = "long-bubble"
            print("ok\n")
            break
        elif target == "2":
            sleep(0.3)
            target = "cross-bubble"
            print("ok\n")
            break
        else:
            sleep(0.3)
            print("\n無効な値です\n")

    config = configparser.ConfigParser()
    config.read("../config/config.ini")
    master_txt_path = config.get('path', 'master')
    target_path     = config.get('path', target)
    target_files    = glob.glob(target_path)
    csv_files       = glob.glob('../results/data/{}/*.csv'.format(target))

    if csv_files == []:
        target_cnt = None
        x_origin = None
        csv_lists = [["Year", "Month", "Day", "Hour", "Minute", "Second", "bubble_position", "movement", "corr_count"]]

        for fname in tqdm(target_files): #決め打ち
            datetime_number, _ = os.path.splitext(os.path.basename(fname))
            created_datetime  = datetime.strptime(str(datetime_number), '%Y%m%d%H%M%S')
            img = trimming(fname)
            identify_scale_list = identify_scale(img, fname)
            scales = identify_scale_list[0]
            if fname == target_files[0]:
                target_cnt, x_origin = identify_bubble(fname, img, identify_scale_list[1])
                corr_list = cross_correlation(fname, img, identify_scale_list[1], target_cnt, 100)
                position = bubble_position(fname, scales, corr_list, x_origin)
                csv_list = [created_datetime.year, created_datetime.month, created_datetime.day, created_datetime.hour, created_datetime.minute, created_datetime.second, position, corr_list[0], corr_list[1]]
                csv_lists.append(csv_list)
            else:
                corr_list = cross_correlation(fname, img, identify_scale_list[1], target_cnt, created_datetime.second)
                if corr_list is not None:
                    position = bubble_position(fname, scales, corr_list, x_origin)
                    csv_list = [created_datetime.year, created_datetime.month, created_datetime.day, created_datetime.hour, created_datetime.minute, created_datetime.second, position, corr_list[0], corr_list[1]]
                    csv_lists.append(csv_list)

        dt_now   = datetime.now().strftime('%Y%m%d%H%M%S')
        csv_path = '../results/data/{}/{}.csv'.format(target, dt_now)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_lists)

        plot.plot(master_txt_path, csv_path, target)

    else:
        csv_file = csv_files[-1]
        plot.plot(master_txt_path, csv_file, target)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

