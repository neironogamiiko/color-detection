import cv2
import cv2 as cv
import numpy as np
import matplotlib as plt

def get_limits(color):
    tmp_color = np.uint8([[color]])
    hsv_color = cv.cvtColor(tmp_color, cv.COLOR_BGR2HSV)

    lower_limit = hsv_color[0][0][0] - 10, 100, 100
    upper_limit = hsv_color[0][0][0] + 10, 255, 255

    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)

    return lower_limit, upper_limit

webcamera = cv.VideoCapture(0)
while True:
    ret, frame = webcamera.read()

    cv.imshow('Frames', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

webcamera.release()
cv.destroyWindow()