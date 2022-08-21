# import cv2
# import numpy as np
# import math
#
# # read input
# img = cv2.imread('test.jpg')
#
# # convert to gray
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # threshold
# thresh = cv2.threshold(gray, 180 , 255, cv2.THRESH_BINARY)[1]
#
# # find largest contour
# contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]
# big_contour = max(contours, key=cv2.contourArea)
#
# # fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree
# ellipse = cv2.fitEllipse(big_contour)
# (xc,yc),(d1,d2),angle = ellipse
# print(xc,yc,d1,d1,angle)
#
# # draw ellipse
# result = img.copy()
# cv2.ellipse(result, ellipse, (0, 255, 0), 3)
#
# # draw circle at center
# xc, yc = ellipse[0]
# cv2.circle(result, (int(xc),int(yc)), 10, (255, 255, 255), -1)
#
# # draw vertical line
# # compute major radius
# rmajor = max(d1,d2)/2
# if angle > 90:
#     angle = angle - 90
# else:
#     angle = angle + 90
# print(angle)
# xtop = xc + math.cos(math.radians(angle))*rmajor
# ytop = yc + math.sin(math.radians(angle))*rmajor
# xbot = xc + math.cos(math.radians(angle+180))*rmajor
# ybot = yc + math.sin(math.radians(angle+180))*rmajor
# cv2.line(result, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 0, 255), 3)
#
# cv2.imwrite("labrador_ellipse.jpg", result)
#
# cv2.imshow("labrador_thresh", thresh)
# cv2.imshow("labrador_ellipse", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()