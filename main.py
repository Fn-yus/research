import cv2
import numpy as np
import extract_needle 
import houghlines 
import identify_scale 
import digitalize

print("======================")
print("Houghlines\n")
print("・針の位置：" + str(houghlines.needle_line))
print("・直線の検出数：" + str(len(houghlines.img_lines)))
print("・調整後の直線数：" + str(len(houghlines.line_list)))
print("・調整後の目盛り座標：" + str(sorted(houghlines.line_list)))
print("・目盛り幅：" + str(np.diff(sorted(houghlines.line_list), n = 1)))
print("・針の座標：" + str(digitalize.main(houghlines.needle_line, sorted(houghlines.line_list))))
print("======================")
print("Identify Bold Scale Along Rows\n")
print("・針の位置：" + str(identify_scale.needle))
print("・目盛り座標：" + str(identify_scale.scales))
print("・目盛り幅：" + str(np.diff(identify_scale.scales, n = 1)))
print("・針の座標：" + str(digitalize.main(identify_scale.needle, identify_scale.scales)))
print("======================")

cv2.waitKey(0)
cv2.destroyAllWindows()



