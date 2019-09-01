import cv2
import extract_needle 
import houghlines 
import identify_scale 

print("======================")
extract_needle.extract_needle()
houghlines.hough_lines()
print("======================")
identify_scale.count_black()
print("======================")

cv2.waitKey(0)
cv2.destroyAllWindows()



