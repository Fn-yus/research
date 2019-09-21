import cv2
import numpy as np
import glob
import os
import configparser
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from tqdm import tqdm

def trimming(fname):
    if "Long-needle" in fname:
        img = cv2.imread(fname)
        height = img.shape[0]
        width = img.shape[1]
        img_trimmed = cv2.rotate(img[192:288, 81:549], cv2.ROTATE_180)
        cv2.imwrite('results/pictures/img.jpg', img_trimmed)
        return img_trimmed     
    else:
        pass

def extract_needle(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGRからHSVに変換
    red_lower = np.array([0, 86, 86]) #下限
    red_upper = np.array([30, 255, 255]) #上限

    img_mask = cv2.inRange(hsv, red_lower, red_upper) #範囲を指定してマスク画像作成
    img_needle = cv2.bitwise_and(img, img, mask=img_mask) #元画像とマスク画像の共通部分を抽出

    return img_needle

def hough_lines(img, img_needle):
    #img = cv2.imread('pictures/sample_1.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #グレースケール化
    img_gray_denoised = cv2.fastNlMeansDenoising(img_gray)
    img_thresh = cv2.adaptiveThreshold(img_gray_denoised,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,5)
    img_canny = cv2.Canny(img_gray_denoised, 50, 150)    
    img_thresh2 = cv2.bitwise_not(img_thresh)
    img_canny2 = cv2.bitwise_not(img_canny)
    #cv2.imwrite('pictures/sample_1_canny.jpg', img_canny2)

    #img_needle = cv2.imread('pictures/sample_1_needle.jpg')
    img_needle_gray = cv2.cvtColor(img_needle, cv2.COLOR_BGR2GRAY)
    ret, img_needle_thresh = cv2.threshold(img_needle_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_needle_thresh2 = cv2.bitwise_not(img_needle_thresh)
    img_needle_thresh_diff = cv2.subtract(img_needle_thresh, img_thresh)
    img_needle_canny = cv2.Canny(img_needle_gray, 50, 150)
    img_needle_canny2 = cv2.bitwise_not(img_needle_canny)
    #cv2.imwrite('pictures/sample_1_needle_canny.jpg', img_needle_thresh_diff)

    needle_lines = cv2.HoughLines(img_needle_thresh2,1,np.pi,20)
    #print(needle_lines)
    needle_list = []
    for needle_line in needle_lines:
        for rho, _ in needle_line:
            needle_list.append(rho)

    needle_line = np.mean(needle_list)

    cv2.line(img, (needle_line, 1000), (needle_line, -1000), (0, 0, 255), 1)

    img_lines = cv2.HoughLines(img_needle_thresh_diff,1,np.pi,30)

    line_list = []
    for line in img_lines:
        for rho, _ in line:
            if line_list == []:
                line_list.append(rho)
            else:
                idx = np.abs(np.asarray(line_list) - rho).argmin()
                if np.abs(line_list[idx] - rho) <= 10:
                    x = np.mean([line_list[idx], rho])
                    line_list.pop(idx)
                    line_list.append(x)
                else:
                    line_list.append(rho)

    for i in line_list:
        cv2.line(img, (i, 1000), (i, -1000), (0, 255, 0), 1)

    cv2.imwrite('sample_pictures/hough_' + fname, img)
    #cv2.imshow('sample_1_canny.jpg', img_canny)
    #cv2.imshow('sample_1_thresh.jpg', img_thresh)
    #cv2.imshow('sample_1_needle_thresh.jpg', img_needle_thresh)
    cv2.imshow('sample_1_needle_thresh_diff.jpg', img_needle_thresh_diff)

    return {"needle_line":needle_line, "img_lines":img_lines, "line_list":line_list, "img_canny":img_canny2, "img_needle_canny":img_needle_canny2}

def identify_scale(img_canny, img_needle_canny, new_fname):
    #img_needle = cv2.imread('pictures/sample_1_needle_canny.jpg')
    #img_needle_gray = cv2.cvtColor(img_needle_canny, cv2.COLOR_BGR2GRAY) #画像を再度読み込んでいるので、再度グレースケール化
    ret, img_needle_thresh = cv2.threshold(img_needle_canny, 127, 255, cv2.THRESH_BINARY)

    needle = []
    for row in img_needle_thresh:
        needle_list, = np.where(row == 0)
        if len(needle_list) == 2:
            if needle_list[0] >= 450 and np.diff(needle_list) >= 3: #端の場合
                needle = np.mean(needle_list)
                break
            elif np.diff(needle_list) >= 5:
                needle = np.mean(needle_list)
                break
        else:
            continue

    if needle == []:
        for row in img_needle_thresh:
            needle_list, = np.where(row == 0) #色が黒の箇所を抽出
            print(len(needle_list))
            print(needle_list)
        print(new_fname)
    else:
        pass

    
    #img = np.array(cv2.imread('pictures/sample_1_canny.jpg'))
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, img_thresh = cv2.threshold(img_canny, 127, 255, cv2.THRESH_BINARY)
    img_thresh_diff = cv2.bitwise_not(cv2.subtract(img_needle_thresh, img_thresh))
    cv2.imwrite('results/pictures/img_needle_thresh.jpg', img_needle_thresh)
    cv2.imwrite('results/pictures/img_thresh.jpg', img_thresh)
    cv2.imwrite('results/pictures/img_thresh_diff.jpg', img_thresh_diff)
    print(needle)

    scale_list = []
    for row in img_thresh_diff:
        black_list, = np.where(row == 0) #色が黒の箇所を抽出
        if len(black_list) == 14 and min(np.diff(black_list[0::2])) > 50: #目盛りの縁が14個かつ目盛り間のピクセル距離が50以上
            scale_list = black_list
            break
        else:
            continue
    
    #うまくいけばここはpass
    if scale_list == []:
        for row in img_thresh_diff:
            black_list, = np.where(row == 0) #色が黒の箇所を抽出
            print(len(black_list))
            print(black_list)
        print(new_fname)
    else:
        pass

    scale_list_splited = np.split(np.array(scale_list), 7)

    scales = []
    for row in scale_list_splited:
        scale = np.mean(row)
        scales.append(scale)
    
    cv2.line(img_canny, (Decimal(str(needle)).quantize(Decimal("0")), 1000), (Decimal(str(needle)).quantize(Decimal("0")), -1000), (0, 0, 255), 1)

    for i in scales:
        cv2.line(img_canny, (Decimal(str(i)).quantize(Decimal("0")), 1000), (Decimal(str(i)).quantize(Decimal("0")), -1000), (0, 255, 0), 1)


    return {"needle":needle, "scales":scales}

def digitalize(needle, scale):
    scale_upper_list, = np.where(needle <= scale)
    scale_upper_index = min(scale_upper_list)
    scale_lower_list, = np.where(needle >= scale)
    scale_lower_index = max(scale_lower_list)
    needle_percentage = (needle - scale[scale_lower_index])/(scale[scale_upper_index] - scale[scale_lower_index])
    needle_position = (scale_lower_index - 3) + needle_percentage
    return needle_position

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config/config.ini")
    files = glob.glob('C:\\Users\\Yusei\\D58-pictures\\Long-needle\\data\\*.jpg') 
    
    for fname in tqdm(files):
        new_fname, ext = os.path.splitext(os.path.basename(fname))
        img = trimming(fname)
        img_needle = extract_needle(img)
        houghlines = hough_lines(img, img_needle)
        identifyscale = identify_scale(houghlines["img_canny"], houghlines["img_needle_canny"], new_fname)
        print("======================")
        print(new_fname)
        print("======================")
        print("Houghlines\n")
        print("・針の位置：" + str(houghlines["needle_line"]))
        print("・直線の検出数：" + str(len(houghlines["img_lines"])))
        print("・調整後の直線数：" + str(len(houghlines["line_list"])))
        print("・調整後の目盛り座標：" + str(sorted(houghlines["line_list"])))
        print("・目盛り幅：" + str(np.diff(sorted(houghlines["line_list"]), n = 1)))
        print("・針の座標：" + str(digitalize(houghlines["needle_line"], sorted(houghlines["line_list"]))))
        print("======================")
        print("Identify Bold Scale Along Rows\n")
        print("・針の位置：" + str(identifyscale["needle"]))
        print("・目盛り座標：" + str(identifyscale["scales"]))
        print("・目盛り幅：" + str(np.diff(identifyscale["scales"], n = 1)))
        print("・針の座標：" + str(digitalize(identifyscale["needle"], identifyscale["scales"])))
        print("======================")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    



