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
from itertools import product

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
    # cv2.imwrite('results/pictures/bubble/img_gray_denoised.jpg', img)
    # cv2.imwrite('results/pictures/bubble/img_bubble_canny.jpg', img_bubble_canny)

    contours, _ = cv2.findContours(img_bubble_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    # with open('results/data/bubble/contours.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(contours)
    cnt_list = None
    for cnt in contours:
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

                if cnt_list == None or cnt_RMS < cnt_list[1]:
                    cnt_list = [cnt, x_cal, cnt_RMS]
                else:
                    continue

    target_cnt = cnt_list[0][:,0] #3次元配列を2次元配列に（[[[a, b]], [[c, d]]] => [[a, b], [c,d]]）
    # print(cnt_list)
    # print(target_cnt)
    # cv2.imwrite('results/pictures/bubble/img_all_contour.jpg', img_all_contour)

    img_ellipse = cv2.drawContours(img_origin, [cnt_list[0]], 0, (255, 0, 0), 1)
    cv2.imwrite('results/pictures/bubble/img_ellipse.jpg', img_ellipse)

    return target_cnt, cnt_list[1]

def cross_correlation(fname, img_origin, img, origin_cnt, created_datetime_second):
    img_bubble_canny = cv2.Canny(img, 250, 550)
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
                if cross_pixel_count != 0: #速くなったりしないかな
                    cross_count += cross_pixel_count
            if correlation_list == None or cross_count > correlation_list[3]:
                correlation_list = [cnt, cross_cnt, x, cross_count]

        datetime_number, _ = os.path.splitext(os.path.basename(fname))

        img_contour = cv2.drawContours(img_origin, [correlation_list[0]], 0, (0, 0, 255), 2)
        cv2.imwrite('results/pictures/contour/img_contour_{}.jpg'.format(datetime_number), img_contour)

        img_wrapped = cv2.drawContours(img_contour, [correlation_list[1]], 0, (255, 0, 0), 1)

        cv2.imwrite('results/pictures/canny/img_canny_{}.jpg'.format(datetime_number), img_bubble_canny)
        cv2.imwrite('results/pictures/wrapped/img_wrapped_{}.jpg'.format(datetime_number), img_wrapped)

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

def plot(master_path, csv_path, target):
    master_data = np.loadtxt(master_path, encoding='utf-8')
    csv_data    = np.loadtxt(csv_path, delimiter=',', skiprows=1, encoding='utf-8')

    csv_schedule = csv_data.astype(int)
    start_time   = datetime(csv_schedule[0,0], csv_schedule[0,1], csv_schedule[0,2], csv_schedule[0,3], csv_schedule[0,4], 00)
    end_time     = datetime(csv_schedule[-1,0], csv_schedule[-1,1], csv_schedule[-1,2], csv_schedule[-1,3], csv_schedule[-1,4], 59)

    sorted_csv_data = []
    for index, csv_row in enumerate(csv_schedule):
        target_time = datetime(csv_row[0], csv_row[1], csv_row[2], csv_row[3], csv_row[4], csv_row[5])
        new_row     = [csv_row[0], csv_row[1], csv_row[2], csv_row[3], csv_row[4], csv_row[5], (target_time - start_time).total_seconds(), csv_data[index,6]]
        sorted_csv_data.append(new_row)
   
    sorted_master_data = []
    experiment_data    = []
    for row in master_data: 
        row_schedule = row.astype(int)
        target_time  = datetime(row_schedule[0], row_schedule[1], row_schedule[2], row_schedule[4], row_schedule[5], row_schedule[6])
        if start_time <= target_time <= end_time:
            if target.lower() == "long-bubble":
                sorted_master_data.append([row_schedule[0], row_schedule[1], row_schedule[2], row_schedule[4], row_schedule[5], row_schedule[6], (target_time - start_time).total_seconds(), row[8]/15.379])
            elif target.lower() == "cross-bubble":
                sorted_master_data.append([row_schedule[0], row_schedule[1], row_schedule[2], row_schedule[4], row_schedule[5], row_schedule[6], (target_time - start_time).total_seconds(), row[9]/17.107])

        experiment_start_time = datetime(2019, 7, 16, 13, 21, 00)
        long_end_time         = datetime(2019, 7, 16, 13, 58, 00)
        cross_start_time      = datetime(2019, 7, 16, 15, 16, 00)
        experiment_end_time   = datetime(2019, 7, 16, 15, 50, 00)
        if experiment_start_time <= target_time <= long_end_time or cross_start_time <= target_time <= experiment_end_time:
            experiment_data.append([(target_time - experiment_start_time).total_seconds(), row[7], row[8]/15.379, row[9]/17.107])
        elif long_end_time <= target_time <= cross_start_time:
            experiment_data.append([(target_time - experiment_start_time).total_seconds(), None, None, None])

    x1 = np.array(sorted_csv_data)[:,6]
    x2 = np.array(sorted_master_data)[:,6]
    y1 = np.array(sorted_csv_data)[:,7]
    y2 = np.array(sorted_master_data)[:,7]
    x3 = []
    y3 = []
    x3_1 = []
    x3_2 = []
    x3_3 = []
    x3_4 = []
    y3_1 = []
    y3_2 = []
    y3_3 = []
    y3_4 = []
    x4 = np.array(experiment_data)[:,0]
    y4_1 = np.array(experiment_data)[:,1]
    y4_2 = np.array(experiment_data)[:,2]
    y4_3 = np.array(experiment_data)[:,3]

    for c_row in sorted_csv_data:
        for m_row in sorted_master_data:
            if c_row[6] == m_row[6]:
                x3.append(c_row[7])
                y3.append(m_row[7])
                if target == "long-bubble":
                    if  0 <= c_row[6] < 660:
                        x3_1.append(c_row[7])
                        y3_1.append(m_row[7])
                    elif 660 <= c_row[6] < 1140:
                        x3_2.append(c_row[7])
                        y3_2.append(m_row[7])
                    elif 1140 <= c_row[6] < 1740:
                        x3_3.append(c_row[7])
                        y3_3.append(m_row[7])
                    elif 1740 <= c_row[6] <= 2220:
                        x3_4.append(c_row[7])
                        y3_4.append(m_row[7])

    r              = np.corrcoef(np.array(x3), np.array(y3))[0,1]
    (a, b, sa, sb) = least_square(np.array(x3), np.array(y3))

    graph_target = target.replace("-", "_")     #返り値例 "long_needle" 
    target_sort  = target.replace(target[-7:], "")  #返り値は "long" か "cross"

    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    fig5 = plt.figure()

    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.scatter(x1, y1)
    ax1.set_title('tilt-{} analog data'.format(graph_target))
    ax1.set_xlabel('t [s]')
    ax1.set_ylabel('tilt-{} value'.format(graph_target))
    ax1.grid(axis='y')

    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.scatter(x2, y2)
    ax2.set_title('tilt-{} digital data'.format(target_sort))
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('tilt-{} value[arc-sec]'.format(target_sort))
    ax2.grid(axis='y')

    ax3 = fig3.add_subplot(1, 1, 1)
    # ax3.scatter(np.array(x3), np.array(y3))
    ax3.scatter(np.array(x3_1), np.array(y3_1), color="pink" ,label="±0 -> +100", alpha=0.5)
    ax3.scatter(np.array(x3_2), np.array(y3_2), color="blue", label="+100 -> ±0", alpha=0.3)
    ax3.scatter(np.array(x3_3), np.array(y3_3), color="purple", label="±0 -> -100", alpha=0.5)
    ax3.scatter(np.array(x3_4), np.array(y3_4), color="orange", label="-100 -> ±0", alpha=0.5)
    # ax3.plot(np.array(x3), (a*np.array(x3)+b), color="red", label="y = ({}±{})x + ({}±{})".format(a, sa, b, sb))
    ax3.legend()
    ax3.set_title('Compare analog data with digital data(r={})'.format(r))
    ax3.set_xlabel('tilt-{} value'.format(graph_target))
    ax3.set_ylabel('tilt-{} value [arc-sec]'.format(target_sort))
    ax3.grid(axis='both')
    
    ax4 = fig4.add_subplot(2, 1, 1)
    ax4.plot(x4, y4_1)
    ax4.set_ylabel('Gravity [mGal]')
    ax4.grid(axis='y')

    ax5 = fig4.add_subplot(2, 1, 2)
    ax5.plot(x4, y4_2)
    ax5.plot(x4, y4_3)
    ax5.set_xlabel('t [s]')
    ax5.set_ylabel('tilt value [arc-sec]')
    ax5.grid(axis='y')
    ax5.legend(['tilt-long', 'tilt-cross'])

    ax6 = fig5.add_subplot(2, 2, 1)
    ax6.scatter(np.array(x3_1), np.array(y3_1), color="pink" ,label="±0 -> +100")
    ax6.set_xlim([-2.5, 2.5])
    ax6.set_ylim([-120, 120])
    ax6.grid(axis='both')
    ax6.legend()

    ax7 = fig5.add_subplot(2, 2, 2)
    ax7.scatter(np.array(x3_2), np.array(y3_2), color="yellow", label="+100 -> ±0")
    ax7.set_xlim([-2.5, 2.5])
    ax7.set_ylim([-120, 120])
    ax7.grid(axis='both')
    ax7.legend()

    ax8 = fig5.add_subplot(2, 2, 3)
    ax8.scatter(np.array(x3_3), np.array(y3_3), color="purple", label="±0 -> -100")
    ax8.set_xlim([-2.5, 2.5])
    ax8.set_ylim([-120, 120])
    ax8.grid(axis='both')
    ax8.legend()

    ax9 = fig5.add_subplot(2, 2, 4)
    ax9.scatter(np.array(x3_4), np.array(y3_4), color="orange", label="-100 -> ±0")
    ax9.set_xlim([-2.5, 2.5])
    ax9.set_ylim([-120, 120]) 
    ax9.grid(axis='both')  
    ax9.legend()

    plt.show()

def least_square(x, y):
    xy = x * y
    square_x = np.square(x)
    square_y = np.square(y)
    N = len(x)

    a = (N * sum(xy) - (sum(x) * sum(y)))/(N * sum(square_x) - (sum(x) ** 2))
    b = (sum(square_x) * sum(y) - sum(x) * sum(xy))/(N * sum(square_x) - (sum(x) ** 2))

    sy = Decimal(((sum(np.square(a * x + b - y)))/(N - 2)) ** 0.5)

    getcontext().prec     = 1
    getcontext().rounding = ROUND_HALF_UP
    sa = sy * Decimal(((N/(N * sum(square_x) - (sum(x) ** 2))) ** 0.5))
    sb = sy * Decimal(((sum(square_x)/(N * sum(square_x) - (sum(x) ** 2))) ** 0.5))

    getcontext().prec = 28
    a = Decimal(a).quantize(sa)
    b = Decimal(b).quantize(sb)

    return float(a), float(b), float(sa), float(sb)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config/config.ini")
    master_txt_path = config.get('path', 'master')
    target_path = config.get('path', 'long-bubble')
    target_files = glob.glob(target_path)
    csv_files       = glob.glob('results/data/{}/*.csv'.format("bubble"))
    if csv_files == []:
        target_cnt = None
        x_origin = None
        csv_lists = [["Year", "Month", "Day", "Hour", "Minute", "Second", "needle_position", "movement", "corr_count"]]

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
                if corr_list != None:
                    position = bubble_position(fname, scales, corr_list, x_origin)
                    csv_list = [created_datetime.year, created_datetime.month, created_datetime.day, created_datetime.hour, created_datetime.minute, created_datetime.second, position, corr_list[0], corr_list[1]]
                    csv_lists.append(csv_list)

        dt_now   = datetime.now().strftime('%Y%m%d%H%M%S')
        csv_path = 'results/data/bubble/{}.csv'.format(dt_now)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_lists)

        plot(master_txt_path, csv_path, "long-bubble")

    else:
        csv_file = csv_files[-1]
        plot(master_txt_path, csv_file, "long-bubble")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

