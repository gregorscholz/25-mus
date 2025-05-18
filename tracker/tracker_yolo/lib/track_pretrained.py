import cv2
from ultralytics import YOLO

model_object = YOLO("yolo11n.pt")
model_pose = YOLO("yolo11n-pose.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result_object = model_object.predict(
        frame,
        classes=[32],
    )[0]
    frame_object = result_object.plot()

    result_pose = model_pose.predict(frame)[0]
    frame_pose = result_pose.plot(boxes=False)

    combined_frame = cv2.addWeighted(frame_object, 0.5, frame_pose, 0.5, 0)

    cv2.imshow("Video", combined_frame)

    key = cv2.waitKey(1)

    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
