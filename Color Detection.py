import cv2 as cv
import numpy as np
from PIL import Image

def get_limits(color):
    tmp_color = np.uint8([[color]]) # the BGR value witch we want to convert to HSV
    hsv_color = cv.cvtColor(tmp_color, cv.COLOR_BGR2HSV)

    lower_limit = hsv_color[0][0][0] - 10, 100, 100
    upper_limit = hsv_color[0][0][0] + 10, 255, 255

    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)

    return lower_limit, upper_limit

color_to_detect = [0,255,255] # yellow in BGR colorspace

webcamera = cv.VideoCapture(0)
while True:
    ret, frame = webcamera.read()
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_limit, upper_limit = get_limits(color_to_detect)
    mask = cv.inRange(hsv_frame, lower_limit, upper_limit)

    PIL_mask = Image.fromarray(mask) # convert from numpy array to PIL
    bounding_box = PIL_mask.getbbox()

    if bounding_box is not None:
        x1, y1, x2, y2 = bounding_box
        cv.rectangle(frame, (x1,y1), (x2, y2), (0,0,255), 5)

    cv.imshow('Frames', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

webcamera.release()
cv.destroyAllWindows()

# Додати можливість розпізнавати декілька кольорів
# Розібратися з waitKey(1). Як параметр у waitKey впливає на якість визначення кольору
# Розібратися як не детектити дуже малі об'єкти