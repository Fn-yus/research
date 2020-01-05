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
        if "long-needle" in fname.lower():
            img_trimmed = cv2.rotate(img[192:288, 60:565], cv2.ROTATE_180)
            cv2.imwrite('results/pictures/needle/img_trimmed.jpg', img_trimmed)
            return img_trimmed
        elif "cross-needle" in fname.lower():
            img_trimmed = img[192:288, 90:535]            
            cv2.imwrite('results/pictures/needle/img_trimmed.jpg', img_trimmed)     
            return img_trimmed
        else:
            pass

def extract_needle(img, fname):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGRからHSVに変換

    if "cross-needle" in fname:
        gamma = 0.8
        lookuptable = np.zeros((256,1),dtype = 'uint8')
        for i in range(256):
            lookuptable[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)

        hsv = cv2.LUT(hsv, lookuptable)

    mask1 = cv2.inRange(hsv, (0, 80, 86), (30, 255, 255))     #赤～黄色に近い赤
    mask2 = cv2.inRange(hsv, (150, 30, 86), (179, 255, 255))  #紫に近い赤～赤

    img_mask = cv2.bitwise_or(mask1, mask2)               #範囲を指定してマスク画像作成
    img_needle = cv2.bitwise_and(img, img, mask=img_mask) #元画像とマスク画像の共通部分を抽出

    cv2.imwrite('results/pictures/needle/img_needle.jpg', img_needle)
    return img_needle  

def identify_scale(img, img_needle, fname):
    img_needle_gray = cv2.cvtColor(img_needle, cv2.COLOR_BGR2GRAY)
    img_needle_denoised = cv2.fastNlMeansDenoising(img_needle_gray)
    img_needle_canny = cv2.Canny(img_needle_denoised, 50, 150)

    img_needle_canny2 = cv2.bitwise_not(img_needle_canny)
    ret, img_needle_thresh = cv2.threshold(img_needle_canny2, 127, 255, cv2.THRESH_BINARY)

    needle = []
    for row in img_needle_thresh:
        needle_list, = np.where(row == 0)
        if len(needle_list) == 2:
            width = img.shape[1]
            if "long-needle" in fname.lower():
                if needle_list[0] >= width - 45 and np.diff(needle_list) >= 3: #端の場合
                    needle.append(np.mean(needle_list))
                    break
                elif np.diff(needle_list) >= 5:
                    needle.append(np.mean(needle_list))
                    break
            if "cross-needle" in fname.lower():
                if needle_list[0] >= width - 25 and np.diff(needle_list) >= 3: #端の場合
                    needle.append(np.mean(needle_list))
                    break
                elif np.diff(needle_list) >= 5:
                    needle.append(np.mean(needle_list))
                    break

        else:
            continue
    
    needle = np.mean(np.array(needle))

    if needle == 0:
        for row in img_needle_thresh:
            needle_list, = np.where(row == 0) #色が黒の箇所を抽出
            print(len(needle_list))
            print(needle_list)
        print(fname)
    else:
        pass

    img_gray          = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #グレースケール化
    img_gray_denoised = cv2.fastNlMeansDenoising(img_gray)
    img_canny         = cv2.Canny(img_gray_denoised, 70, 150)
    img_canny2        = cv2.bitwise_not(img_canny)
    _, img_thresh     = cv2.threshold(img_canny2, 127, 255, cv2.THRESH_BINARY)

    if "long-needle" in fname.lower():
        img_thresh[-1000:1000, 0:20]    = 255  #画像左端の映り込み部分を削除
        img_thresh[-1000:1000, 484:505] = 255  #画像右端の映り込み部分を削除
    elif "cross-needle" in fname.lower():
        img_thresh[-1000:1000, 0:15]    = 255  #画像左端の影映り込み部分を削除

    scale_list = []
    for row in img_thresh:
        black_list, = np.where(row == 0) #色が黒の箇所を抽出
        if "needle" in fname.lower():
            if len(black_list) == 14 and min(np.diff(black_list[0::2])) > 50: #目盛りの縁が14個かつ目盛り間のピクセル距離が50以上
                scale_list.append(black_list)
    
    #うまくいけばここはpass
    if len(scale_list) == 0:
        for row in img_thresh:
            black_list, = np.where(row == 0) #色が黒の箇所を抽出
            print(len(black_list))
            print(black_list)
        print(fname)
    else:
        pass

    scale_list         = np.mean(np.array(scale_list), axis=0)      #条件に一致する線の平均をリストに
    scale_list_splited = np.split(np.array(scale_list), 7)  #目盛りの左右の線ごとにまとめる
    scales             = [np.mean(row) for row in scale_list_splited]   #目盛りの左右の線の平均を取り、scalesにappend

    cv2.line(img, (Decimal(str(needle)).quantize(Decimal("0")), 1000), (Decimal(str(needle)).quantize(Decimal("0")), -1000), (0, 0, 255), 1)
    for i in scales:
        cv2.line(img, (Decimal(str(i)).quantize(Decimal("0")), 1000), (Decimal(str(i)).quantize(Decimal("0")), -1000), (0, 255, 0), 1)

    cv2.imwrite('results/pictures/needle/img.jpg', img)
    cv2.imwrite('results/pictures/needle/img_needle_thresh.jpg', img_needle_thresh)
    cv2.imwrite('results/pictures/needle/img_thresh.jpg', img_thresh)

    return needle, scales

def digitalize(needle, scales):
    if needle in scales: #針が目盛りと完全一致した場合
        needle_position = -3 + scales.index(needle)
        return needle_position
    else:
        scale_upper_list, = np.where(needle <= scales)
        scale_lower_list, = np.where(needle >= scales)

        if len(scale_upper_list)   == 0: #一番右の目盛りより右側に針がある場合
            return None

        elif len(scale_lower_list) == 0: #一番左の目盛りより左側に針がある場合
            return None

        else:
            scale_upper       = scales[min(scale_upper_list)]

            scale_lower_index = max(scale_lower_list)
            scale_lower       = scales[scale_lower_index]

            needle_percentage = (needle - scale_lower)/(scale_upper - scale_lower)
            needle_position   = (scale_lower_index - 3) + needle_percentage

            return needle_position

def plot(master_file_path, csv_file_path, target):
    master_data = np.loadtxt(master_file_path, encoding='utf-8')
    csv_data    = np.loadtxt(csv_file_path, delimiter=',', skiprows=1, encoding='utf-8')

    csv_schedule = csv_data.astype(int)
    start_time   = datetime(csv_schedule[0,0], csv_schedule[0,1], csv_schedule[0,2], csv_schedule[0,3], csv_schedule[0,4], 00)
    end_time     = datetime(csv_schedule[-1,0], csv_schedule[-1,1], csv_schedule[-1,2], csv_schedule[-1,3], csv_schedule[-1,4], 59)

    sorted_csv_data = []
    for index, csv_row in enumerate(csv_schedule):
        target_time = datetime(csv_row[0], csv_row[1], csv_row[2], csv_row[3], csv_row[4], csv_row[5])
        new_row     = [csv_row[0], csv_row[1], csv_row[2], csv_row[3], csv_row[4], csv_row[5], (target_time - start_time).total_seconds(), csv_data[index,14]]
        sorted_csv_data.append(new_row)
   
    sorted_master_data = []
    experiment_data    = []
    for row in master_data: 
        row_schedule = row.astype(int)
        target_time  = datetime(row_schedule[0], row_schedule[1], row_schedule[2], row_schedule[4], row_schedule[5], row_schedule[6])
        if start_time <= target_time <= end_time:
            if target.lower() == "long-needle":
                sorted_master_data.append([row_schedule[0], row_schedule[1], row_schedule[2], row_schedule[4], row_schedule[5], row_schedule[6], (target_time - start_time).total_seconds(), row[8]/15.379])
            elif target.lower() == "cross-needle":
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
    x4 = np.array(experiment_data)[:,0]
    y4_1 = np.array(experiment_data)[:,1]
    y4_2 = np.array(experiment_data)[:,2]
    y4_3 = np.array(experiment_data)[:,3]

    for c_row in sorted_csv_data:
        for m_row in sorted_master_data:
            if c_row[6] == m_row[6]:
                x3.append(c_row[7])
                y3.append(m_row[7])

    r              = np.corrcoef(np.array(x3), np.array(y3))[0,1]
    (a, b, sa, sb) = least_square(np.array(x3), np.array(y3))

    graph_target = target.replace("-", "_")        #返り値例 "long_needle" 
    target_sort  = target.replace(target[-7:], "")  #返り値は "long" か "cross"

    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()

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
    ax3.scatter(np.array(x3), np.array(y3))
    ax3.plot(np.array(x3), (a*np.array(x3)+b), color="red")
    ax3.set_title('Compare about analog and digital data(r={})'.format(r))
    ax3.set_xlabel('tilt-{} value'.format(graph_target))
    ax3.set_ylabel('tilt-{} value [arc-sec]'.format(target_sort))
    ax3.grid(axis='both')
    ax3.legend(["y = ({}±{})x + ({}±{})".format(a, sa, b, sb)])

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
    target = None
    while True:
        target = input("解析する画像の種類を選んでください\n[1.long-needle, 2.cross-needle]：")
        if target.lower() == "long-needle" or target.lower() == "cross-needle":
            sleep(0.3)
            print("ok\n")
            break
        elif target == "1":
            sleep(0.3)
            target = "long-needle"
            print("ok\n")
            break
        elif target == "2":
            sleep(0.3)
            target = "cross-needle"
            print("ok\n")
            break
        else:
            sleep(0.3)
            print("\n無効な値です\n")

    config = configparser.ConfigParser()
    config.read("config/config.ini")
    master_txt_path = config.get('path', 'master')
    target_path     = config.get('path', target)
    target_files    = glob.glob(target_path)
    csv_files       = glob.glob('results/data/{}/*.csv'.format(target))

    if  csv_files == []:
        print("画像を解析しています...")

        csv_lists = [["Year", "Month", "Day", "Hour", "Minute", "Second", "Scale:-3", "Scale:-2", "Scale:-1", "Scale:0", "Scale:1", "Scale:2", "Scale:3", "Needle", "NeedleValue"]]
        
        for fname in tqdm(target_files):
            datetime_number, _ = os.path.splitext(os.path.basename(fname))
            created_datetime  = datetime.strptime(str(datetime_number), '%Y%m%d%H%M%S')
            img               = trimming(fname)
            img_needle        = extract_needle(img, fname)
            identifyscale     = identify_scale(img, img_needle, fname)
            needle_position   = digitalize(identifyscale[0], identifyscale[1])
            if needle_position is not None and created_datetime.second >= 10:
                csv_list = sum([[created_datetime.year, created_datetime.month, created_datetime.day, created_datetime.hour, created_datetime.minute, created_datetime.second], identifyscale[1], [identifyscale[0], needle_position]], []) #平坦化している
                csv_lists.append(csv_list)
        
        dt_now   = datetime.now().strftime('%Y%m%d%H%M%S')
        csv_path = 'results/data/{}/{}.csv'.format(target, dt_now)
        with open(csv_path, 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerows(csv_lists)
        print("\ncsvファイルが作成されました\nグラフを作成しています...")
        plot(master_txt_path, csv_path, target)

    else:
        print("csvファイルが既に存在しています\nグラフを作成しています...")
        csv_path = csv_files[-1]
        plot(master_txt_path, csv_path, target)
