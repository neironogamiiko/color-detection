import cv2
import cv2 as cv
import numpy as np
import PIL
from PIL.Image import Image


def get_limits(color):
    tmp_color = np.uint8([[color]]) # the BGR value witch we want to convert to HSV
    hsv_color = cv.cvtColor(tmp_color, cv.COLOR_BGR2HSV)

    lower_limit = hsv_color[0][0][0] - 10, 100, 100
    upper_limit = hsv_color[0][0][0] + 10, 255, 255

    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)

    return lower_limit, upper_limit

color_to_detect = [0,255,255] # red in BGR colorspace

webcamera = cv.VideoCapture(0)
while True:
    ret, frame = webcamera.read()
    hsv_frame = cv.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_limit, upper_limit = get_limits(color_to_detect)
    mask = cv.inRange(hsv_frame, lower_limit, upper_limit)

    cv.imshow('Frames', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcamera.release()
cv.destroyAllWindows()