import cv2
import numpy as np 

img_needle = cv2.imread('pictures/sample_1_needle_canny.jpg')
img_needle_gray = cv2.cvtColor(img_needle, cv2.COLOR_BGR2GRAY) #画像を再度読み込んでいるので、再度グレースケール化
ret, img_needle_thresh = cv2.threshold(img_needle_gray, 127, 255, cv2.THRESH_BINARY)

needle = None
for row in img_needle_thresh:
    needle_list, = np.where(row == 0)
    if len(needle_list) == 2 and np.diff(needle_list) > 5:
        needle = np.mean(needle_list)
        break 
    else:
        continue

img = np.array(cv2.imread('pictures/sample_1_canny.jpg'))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
ret, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

scale_list = None
for row in img_thresh:
    black_list, = np.where(row == 0)
    if len(black_list) == 14 and min(np.diff(black_list[0::2])) > 50: #目盛りの縁が14個かつ目盛り間のピクセル距離が50以上
        scale_list = black_list
        break
    else:
        continue

scale_list_splited = np.split(np.array(scale_list), 7)

scales = []
for row in scale_list_splited:
    scale = np.mean(row)
    scales.append(scale)
