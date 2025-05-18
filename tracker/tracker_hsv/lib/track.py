import math

import cv2
import numpy as np

RED = [
    "RED",
    np.array([0, 27, 217], np.uint8),
    np.array([7, 206, 255], np.uint8),
]
RED1 = [
    "RED1",
    np.array([0, 100, 100], np.uint8),
    np.array([10, 255, 255], np.uint8),
]
RED2 = [
    "RED2",
    np.array([160, 100, 100], np.uint8),
    np.array([179, 255, 255], np.uint8),
]
BLUE = [
    "BLUE",
    np.array([70, 65, 205], np.uint8),
    np.array([140, 255, 255], np.uint8),
]
YELLOW = [
    "YELLOW",
    np.array([20, 100, 100], np.uint8),
    np.array([30, 255, 255], np.uint8),
]


def build_mask_red(frame):
    blur = cv2.GaussianBlur(frame, (0, 0), 3)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask0 = cv2.inRange(hsv, RED[1], RED[2])
    mask1 = cv2.inRange(hsv, RED1[1], RED1[2])
    mask2 = cv2.inRange(hsv, RED2[1], RED2[2])

    return cv2.bitwise_or(mask0, mask1, mask2)


def build_mask(frame, color):
    blur = cv2.GaussianBlur(frame, (0, 0), 3)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = None
    if color is RED:
        mask = build_mask_red(frame)
    else:
        mask = cv2.inRange(hsv, color[1], color[2])

    return mask


def filter_contours(contours):
    round_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = 4 * math.pi * (area / (perimeter * perimeter))

        if circularity > 0.7:
            round_contours.append(contour)
    return round_contours


def draw_ball(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_contours(contours)

    if not contours:
        return

    max_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(max_contour) > 70:
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (255, 255, 255), 2, lineType=cv2.LINE_AA)


def track_color(frame, masks):
    for mask in masks:
        m = build_mask(frame, mask)
        cv2.imshow(f"HSV {mask[0]}", m)
        draw_ball(m)
    return frame


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = track_color(frame, [RED, BLUE, YELLOW])

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
