import cv2 
import numpy as np

def hough_lines():
    img = cv2.imread('pictures/sample_1.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #グレースケール化
    img_gray_denoised = cv2.fastNlMeansDenoising(img_gray)
    ret, img_thresh = cv2.threshold(img_gray_denoised, 127, 255, cv2.THRESH_BINARY) #二値化

    img_needle = cv2.imread('pictures/sample_1_needle.jpg')
    img_needle_gray = cv2.cvtColor(img_needle, cv2.COLOR_BGR2GRAY)
    ret, img_needle_thresh = cv2.threshold(img_needle_gray, 50, 255, cv2.THRESH_BINARY_INV)

    img_neddle_thresh_diff = cv2.subtract(img_thresh, img_needle_thresh)
    
    cv2.imshow('sample_1_thresh.jpg', img_thresh)
    cv2.imshow('sample_1_needle_thresh.jpg', img_needle_thresh)
    cv2.imshow('sample_1_needle_thresh_diff.jpg', img_needle_thresh)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    hough_lines()

