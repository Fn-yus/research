import cv2
import numpy as np 

def count_black():
    img = np.array(cv2.imread('pictures/sample_1_canny.jpg'))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #画像を再度読み込んでいるので、再度グレースケール化
    ret, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    scale_lists = []
    for row in img_thresh:
       black_list, = np.where(row == 0)
       if len(black_list) == 14:
           scale_lists.append(black_list)
    
    print(scale_lists)

    scale_list_not_flatten = []
    for row in scale_lists:
        print(row[0::2])
        if min(np.diff(row[0::2])) > 30: #目盛りの差が30ピクセル以上の時のみパス
            scale_list_not_flatten.append(row)
            break
    
    scale_list = np.split(np.array(scale_list_not_flatten[0]), 7)
    
    scales = []
    for row in scale_list:
        scale = np.mean(row)
        scales.append(scale)

    print(scales)
    print(np.diff(scales))
    
    

    

    
        
       
    
           








if __name__ == '__main__':
    count_black()
