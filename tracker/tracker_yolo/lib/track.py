import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")


def detect(frame):
    results = model(frame)
    return results


def highlight(frame, results):

    for r in results:
        print(r.boxes)

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

    return frame


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(frame, [256, 256])

    results = detect(frame)

    frame = highlight(frame, results)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Video", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
