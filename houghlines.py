import cv2 
import numpy as np

def hough_lines():
    img = cv2.imread('pictures/sample_1.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #グレースケール化
    img_gray_denoised = cv2.fastNlMeansDenoising(img_gray)
    img_thresh = cv2.adaptiveThreshold(img_gray_denoised,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,5)
    img_canny = cv2.Canny(img_gray_denoised, 50, 150)    
    img_thresh2 = cv2.bitwise_not(img_thresh)
    img_canny2 = cv2.bitwise_not(img_canny)
    cv2.imwrite('pictures/sample_1_canny.jpg', img_canny2)

    img_needle = cv2.imread('pictures/sample_1_needle.jpg')
    img_needle_gray = cv2.cvtColor(img_needle, cv2.COLOR_BGR2GRAY)
    ret, img_needle_thresh = cv2.threshold(img_needle_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_needle_thresh2 = cv2.bitwise_not(img_needle_thresh)
    img_needle_thresh_diff = cv2.subtract(img_needle_thresh, img_thresh)

    needle_lines = cv2.HoughLines(img_needle_thresh2,1,np.pi,20)
    #print(needle_lines)
    needle_list = []
    for needle_line in needle_lines:
        for rho, _ in needle_line:
            needle_list.append(rho)
    
    needle_line = np.mean(needle_list)
    print("・針の座標：" + str(needle_line))
    cv2.line(img, (needle_line, 1000), (needle_line, -1000), (0, 0, 255), 1)

    img_lines = cv2.HoughLines(img_needle_thresh_diff,1,np.pi,30)
    print("・直線の検出数：" + str(len(img_lines)))
    print_lines(img, img_lines)

    cv2.imshow('sample_1_hough.jpg', img)
    #cv2.imshow('sample_1_canny.jpg', img_canny)
    #cv2.imshow('sample_1_thresh.jpg', img_thresh)
    #cv2.imshow('sample_1_needle_thresh.jpg', img_needle_thresh)
    cv2.imshow('sample_1_needle_thresh_diff.jpg', img_needle_thresh_diff)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_lines(img, lines):
    line_list = []
    for line in lines:
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
    
    print("・調整後の直線数：" + str(len(line_list)))
    print("・調整後の目盛り座標：" + str(sorted(line_list)))
    print("・目盛り幅：" + str(np.diff(sorted(line_list))))

if __name__ == '__main__':
    hough_lines()

