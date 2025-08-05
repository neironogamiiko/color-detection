import cv2 as cv
import numpy as np

def get_limits(color):
    tmp_color = np.uint8([[color]])  # BGR to HSV
    hsv_color = cv.cvtColor(tmp_color, cv.COLOR_BGR2HSV)

    lower_limit = hsv_color[0][0][0] - 10, 100, 100 # upper limit for color value
    upper_limit = hsv_color[0][0][0] + 10, 255, 255 # lwoer limit for color value

    lower_limit = np.array(lower_limit, dtype=np.uint8) # lower limit -10 hue range
    upper_limit = np.array(upper_limit, dtype=np.uint8) # upper limit +10 hue range

    return lower_limit, upper_limit

color_to_detect = [0, 255, 255]  # Yellow in BGR

webcamera = cv.VideoCapture(0)

while True:
    ret, frame = webcamera.read()

    if not ret:
        break

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_limit, upper_limit = get_limits(color_to_detect)
    mask = cv.inRange(hsv_frame, lower_limit, upper_limit)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv.contourArea(contour)
        if area > 150: # ignore small contours
            x, y, width, height = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

    cv.imshow('Frames', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

webcamera.release()
cv.destroyAllWindows()

# To-Do List:
# Зробити вибір кольору
# Зробити більші межі кольору
