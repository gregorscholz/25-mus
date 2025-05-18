import string

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


def draw_bbox(image, bound_ball_pair, pair_ball):
    image_h, image_w = image.shape[:2]

    # draw bbox on ball
    for ball in pair_ball:
        if ball["frequency"] >= 2:
            continue
        cv2.rectangle(image, ball["p1"], ball["p2"], (255, 0, 0), 3)
        text = "ball " + str(ball["ID"]) + " " + str(ball["state"])
        cv2.putText(
            image,
            text,
            (int(ball["p1"][0]), int(ball["p1"][1]) + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            int(0.6 * (image_h + image_w) / 600),
        )

    return image


def track(frame, bbox):
    centroids = []
    num_classes = 1
    image_h, image_w, _ = frame.shape

    out_boxes, out_scores, out_classes, num_boxes = bbox
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes:
            continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)
        c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
        width = int(c2[0] - c1[0])
        height = int(c2[1] - c1[1])
        cw = c1[0] + (width / 2)
        ch = c1[1] + (height / 2)

        # check distance level with recpect to ball width ratio
        distance_level = 1 if width >= 57 else 2
        # centroids.append([cw, ch, c1, c2, distance_level, string.ascii_lowercase[i]])
        centroids.append(
            {
                "centroid": [cw, ch],
                "p1": c1,
                "p2": c2,
                "distance_level": distance_level,
                "ID": string.ascii_lowercase[i],
                "state": "unbound",  # Adding a default state
                "frequency": 0,
            }
        )

    return centroids


def track_ball(frame):
    image_data = cv2.resize(frame, (416, 416))
    image_data = image_data / 255.0
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # capture the detection box
    batch_data = tf.constant(image_data)
    pred_bbox_ball = infer_ball(batch_data)
    for _, value in pred_bbox_ball.items():
        boxes_ball = value[:, :, 0:4]
        pred_conf_ball = value[:, :, 4:]

        boxes, scores, classes, valid_detections = (
            tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes_ball, (boxes_ball.shape[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf_ball,
                    (pred_conf_ball.shape[0], -1, pred_conf_ball.shape[-1]),
                ),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.25,
                score_threshold=0.30,
            )
        )

        pred_bbox = [
            boxes.numpy(),
            scores.numpy(),
            classes.numpy(),
            valid_detections.numpy(),
        ]

        ball = track(frame, pred_bbox)
        frame_with_boxes = draw_bbox(frame, ball, ball)

        if frame_with_boxes is None:
            return frame

        return frame_with_boxes


saved_model_loaded_ball = tf.saved_model.load(
    "ball_weights", tags=[tag_constants.SERVING]
)
infer_ball = saved_model_loaded_ball.signatures["serving_default"]  # type: ignore


# cap = cv2.VideoCapture("videos/video1.mp4")
cap = cv2.VideoCapture(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (int(width), int(height)))

    frame = track_ball(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
