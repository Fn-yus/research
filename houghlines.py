import cv2 
import numpy as np

def hough_lines():
    img = cv2.imread('pictures/sample_1.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #グレースケール化
    cv2.imshow("img_gray.jpg", img_gray)
    img_gray_denoised = cv2.fastNlMeansDenoising(img_gray)
    cv2.imshow("img_gray_denoised.jpg", img_gray_denoised)
    #ret, img_thresh = cv2.threshold(img_gray_denoised, 150, 255, cv2.THRESH_BINARY) #二値化
    img_thresh = cv2.adaptiveThreshold(img_gray_denoised,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    img_canny = cv2.Canny(img_gray_denoised, 50, 150)    
    img_thresh2 = cv2.bitwise_not(img_thresh)
    img_canny2 = cv2.bitwise_not(img_canny)
    

    #contours, hierarchy = cv2.findContours(img_thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE) #輪郭の点検出
    #for i, contour in enumerate(contours):
    #    x,y,w,h = cv2.boundingRect(contour)
    #    img_contours = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #cv2.imshow('sample_1_contours.jpg', img_contours)

    img_needle = cv2.imread('pictures/sample_1_needle.jpg')
    img_needle_gray = cv2.cvtColor(img_needle, cv2.COLOR_BGR2GRAY)
    ret, img_needle_thresh = cv2.threshold(img_needle_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_needle_thresh2 = cv2.bitwise_not(img_needle_thresh)
    img_needle_thresh_diff = cv2.subtract(img_needle_thresh, img_thresh)
    img_canny_diff = cv2.subtract(img_needle_thresh, img_canny)
    cv2.imshow("img_canny_diff.jpg", img_canny_diff)


    needle_lines = cv2.HoughLines(img_needle_thresh2,1,np.pi,20)
    print(needle_lines)
    print(len(needle_lines))
    for needle_line in needle_lines:
        for rho,theta in needle_line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        print((x1, y1), (x2, y2))

    img_lines = cv2.HoughLines(img_needle_thresh_diff,1,np.pi,32)
    print(len(img_lines))
    for img_line in img_lines:
        print(img_line)
        for rho,theta in img_line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


    cv2.imshow('sample_1_hough.jpg', img)
    #cv2.imshow('sample_1_canny.jpg', img_canny)
    cv2.imshow('sample_1_thresh.jpg', img_thresh)
    cv2.imshow('sample_1_needle_thresh.jpg', img_needle_thresh)
    cv2.imshow('sample_1_needle_thresh_diff.jpg', img_needle_thresh_diff)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    hough_lines()

