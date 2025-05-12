import cv2
import numpy as np

RED = [np.array([0, 27, 217]), np.array([7, 206, 255])]
RED1 = [np.array([0, 100, 100]), np.array([10, 255, 255])]
RED2 = [np.array([160, 100, 100]), np.array([179, 255, 255])]
BLUE = [np.array([70, 65, 205]), np.array([140, 255, 255])]
YELLOW = [np.array([20, 100, 100]), np.array([30, 255, 255])]


def build_mask_red(frame):
    blur = cv2.GaussianBlur(frame, (0, 0), 3)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask0 = cv2.inRange(hsv, RED[0], RED[1])
    mask1 = cv2.inRange(hsv, RED1[0], RED1[1])
    mask2 = cv2.inRange(hsv, RED2[0], RED2[1])

    return cv2.bitwise_or(mask0, mask1, mask2)


def build_mask(frame, color):
    if color is RED:
        return build_mask_red(frame)

    frame = cv2.resize(frame, (640, 360))
    blur = cv2.GaussianBlur(frame, (0, 0), 3)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, color[0], color[1])


def draw_ball(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return

    max_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(max_contour) > 70:
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (255, 255, 255), 2, lineType=cv2.LINE_AA)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    red_mask = build_mask(frame, RED)
    draw_ball(red_mask)

    yellow_mask = build_mask(frame, YELLOW)
    draw_ball(yellow_mask)

    blue_mask = build_mask(frame, BLUE)
    draw_ball(blue_mask)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
