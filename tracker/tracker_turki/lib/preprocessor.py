from functools import reduce

import cv2
import numpy as np

YELLOW = [
    np.array([10, 100, 50], np.uint8),
    np.array([40, 255, 255], np.uint8),
]

BLUE = [
    np.array([80, 50, 50], np.uint8),
    np.array([130, 255, 255], np.uint8),
]

RED1 = [
    np.array([0, 100, 80], np.uint8),
    np.array([10, 255, 255], np.uint8),
]
RED2 = [
    np.array([160, 100, 80], np.uint8),
    np.array([179, 255, 255], np.uint8),
]


def filter_balls(frame):
    masks = []
    for color in [YELLOW, BLUE, RED1, RED2]:
        masks.append(_filter_color(frame, color))

    mask = reduce(cv2.bitwise_or, masks)
    return cv2.bitwise_not(mask)


def _filter_color(frame, color):
    blur = cv2.GaussianBlur(frame, (0, 0), 3)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    return cv2.inRange(hsv, color[0], color[1])
