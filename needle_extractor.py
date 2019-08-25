import cv2
import numpy as np

def extract_needle():
    img = cv2.imread("sample_1.jpg") #画像読み込み

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGRからHSVに変換
    red_lower = np.array([0, 100, 100]) #下限
    red_upper = np.array([30, 255, 255]) #上限

    img_mask = cv2.inRange(hsv, red_lower, red_upper) #範囲を指定してマスク画像作成
    img_color = cv2.bitwise_and(img, img, mask=img_mask) #元画像とマスク画像の共通部分を抽出

    cv2.imwrite("sample_1_hsv.jpg", img_color) #画像書き出し

    #img_diff = cv2.add(img, img_color)
    #cv2.imwrite("sample_1_hsv_diff.jpg", img_diff)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
if __name__ == '__main__':
    extract_needle()
    



