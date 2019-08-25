import cv2 
import numpy as np

def hough_lines():
    img = cv2.imread('pictures/sample_1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #グレースケール化
    ret, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) #二値化

    cv2.imshow('sample_1_threshold.jpg', threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    hough_lines()

