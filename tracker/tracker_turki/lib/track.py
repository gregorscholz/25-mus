import csv
import os
import string

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

RADIUS_COLOR = 3
THRESHOLD_COLOR = 60

ball_colors = {}
no_colors_tracked = True

def identify_balls(frame, balls):
    global ball_colors
    global no_colors_tracked

    if no_colors_tracked:
        # check for 3 balls in frame
        i = 0
        for b in balls:
            if b["frequency"] < 2:
                i += 1
        if i < 3:
            return balls

    for i, ball in enumerate(balls):
        if ball["frequency"] >= 2:
            continue

        colors = []
        x = ball['centroid'][0]
        y = ball['centroid'][1]

        for dy in range(-RADIUS_COLOR, RADIUS_COLOR + 1):
            for dx in range(-RADIUS_COLOR, RADIUS_COLOR + 1):
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < frame.shape[1] and 0 <= ny < frame.shape[0]:
                    color = frame[int(ny), int(nx)]
                    colors.append(color)

        if colors:
            avg_color = np.mean(colors, axis=0)
        else:
            continue

        if not no_colors_tracked:
            avg_color = np.mean(colors, axis=0)

            minID = "d"
            minDistance = 100000

            # check for fitting color
            for k, v in ball_colors.items():
                distance = np.linalg.norm(avg_color - v)
                if distance < minDistance:
                    minDistance = distance
                    minID = k

            if minDistance >= THRESHOLD_COLOR:
                ball["ID"] = "d"
                continue
            ball["ID"] = minID
        else:
            bid = string.ascii_lowercase[i]
            ball["ID"] = bid

            ball_colors[bid] = np.mean(colors, axis=0)
            i += 1

    no_colors_tracked = False
    return balls


def draw_bbox(image, bound_ball_pair, pair_ball):
    image_h, image_w = image.shape[:2]

    for ball in pair_ball:
        if ball["frequency"] >= 2:
            continue

        recColor = (0,0,0)
        match ball["ID"]:
            case "a": recColor = (255,0,0)
            case "b": recColor = (0,255,0)
            case "c": recColor = (0,0,255)

        cv2.rectangle(image, ball["p1"], ball["p2"], recColor, 3)
        text = str(ball["ID"])
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
        ball = identify_balls(frame, ball)
        frame_with_boxes = draw_bbox(frame, ball, ball)

        if frame_with_boxes is None:
            return frame

        return frame_with_boxes, ball


saved_model_loaded_ball = tf.saved_model.load(
    "ball_weights", tags=[tag_constants.SERVING]
)
infer_ball = saved_model_loaded_ball.signatures["serving_default"]  # type: ignore


# cap = cv2.VideoCapture(0)

print(os.listdir("videos/"))
for video_name in os.listdir("videos"):
    if str (video_name).endswith(".mp4"):
        cap = cv2.VideoCapture(f"videos/{video_name}")

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        current_frame_counter = 1
        ball_colors = {}
        no_colors_tracked = True

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video processing complete")
                break

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (int(width), int(height)))

            frame, balls = track_ball(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Video", frame)

            with open("test.csv", "a", newline='') as fp:
                writer = csv.writer(fp)
                for b in balls:
                        l = [video_name, current_frame_counter, b["centroid"], b["ID"]]
                        writer.writerow(l)
            current_frame_counter += 1

            if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
