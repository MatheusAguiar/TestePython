import cv2
import time
import matplotlib.pyplot as plt
from imutils.video import VideoStream

cap = VideoStream(src=0).start()
time.sleep(1.0)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

bbox = cv2.selectROI(frame, False)

x, y, w, h = bbox
track_window = (x, y, w, h)

roi = frame[y:y+h, x:x+w]
cv2.imshow('ROI', roi)
#cv2.waitKey(0)

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
cv2.imshow("ROI HSV", hsv_roi)
#cv2.waitKey(0)


roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

plt.hist(roi.ravel(), 180, [0, 180])
plt.show()
cv2.waitKey(0)

roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

