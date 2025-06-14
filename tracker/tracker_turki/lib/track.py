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
balls_last_known_locations = {}
balls_locations = {}

def find_closest_previous_ball_and_update_ball_colors(frame, ball):
    # find closest ball

    # temporary
    avg_color = get_avg_ball_color(frame, ball)

    if avg_color is None:
        return ball

    ck = "d"
    minCDistance = 100000
    sck = "d"

    # check for fitting color
    for k, v in ball_colors.items():
        cDistance = np.linalg.norm(avg_color - v)
        if cDistance < minCDistance:
            minCDistance = cDistance
            sck = ck
            ck = k

    dk = "d"
    minDistance = 100000
    # check for shortest distance
    for k, v in balls_last_known_locations.items():
        distance = np.linalg.norm(np.array(v) - np.array(ball["centroid"]), axis=0)
        if distance < minDistance:
            minDistance = distance
            dk = k

    # check distance supports fitting color
    if dk != ck:
        ball["ID"] = "d"
        return ball
    else: ball["ID"] = dk

    # update ball_colors
    weights = [0.95, 0.05]
    bid = ball["ID"]

    ball_colors[bid] = np.average([ball_colors[bid], avg_color], weights=weights, axis=0)

    return ball


def check_balls(frame, balls):
    # check for identification
    counter = {"a": 0, "b": 0, "c": 0}
    for ball in balls:
        match ball["ID"]:
            case "a": 
                counter["a"] += 1
            case "b": 
                counter["b"] += 1
            case "c": 
                counter["c"] += 1
            case "d": 
                ball = find_closest_previous_ball_and_update_ball_colors(frame, ball)
                # update counter
                if ball['ID'] == "a": counter["a"] += 1
                elif ball['ID'] == "b": counter["b"] += 1
                else: counter["b"] += 1

    # multiple balls of one color
    for key, value in counter.items():
        if value > 1:
            for ball in balls:
                if ball['ID'] == key:
                    ball = find_closest_previous_ball_and_update_ball_colors(frame, ball)

    return balls

def get_avg_ball_color(frame, ball):
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
        return np.mean(colors, axis=0) # avg_color
    else:
        return None


def identify_balls(frame, balls):
    global ball_colors
    global no_colors_tracked

    if no_colors_tracked:
        # check for 3 balls in frame
        if len(balls) != 3:
            return balls

    for i, ball in enumerate(balls):
        avg_color = get_avg_ball_color(frame, ball)

        if avg_color is None:
            continue

        if not no_colors_tracked:
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

            ball_colors[bid] = avg_color
            i += 1

    no_colors_tracked = False
    return balls


def draw_bbox(image, balls):
    image_h, image_w = image.shape[:2]

    for ball in balls:
        recColor = (0,0,0)
        match ball["ID"]:
            case "a": recColor = (255,0,0)
            case "b": recColor = (0,255,0)
            case "c": recColor = (0,0,255)

        cv2.rectangle(image, ball["p1"], ball["p2"], recColor, 3)
        cv2.putText(
            image,
            "",
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
            }
        )

    return centroids


def track_balls(frame):
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

        balls = track(frame, pred_bbox)
        balls = identify_balls(frame, balls)
        balls = check_balls(frame, balls)
        frame_with_boxes = draw_bbox(frame, balls)

        if frame_with_boxes is None:
            return frame

        return frame_with_boxes, balls


def get_last_coordinates_and_number_of_emptys(list):
    counter = 0
    while list[counter+1] == [0,0]:
        counter +=1
    return list[counter+1], counter


saved_model_loaded_ball = tf.saved_model.load(
    "ball_weights", tags=[tag_constants.SERVING]
)
infer_ball = saved_model_loaded_ball.signatures["serving_default"]  # type: ignore


print(os.listdir("videos/"))
for video_name in os.listdir("videos"):
    if str (video_name).endswith(".mp4"):
        cap = cv2.VideoCapture(f"videos/{video_name}")

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        current_frame_counter = 1
        ball_colors = {}
        no_colors_tracked = True
        balls_last_known_locations = {}
        balls_locations = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video processing complete")
                break

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (int(width), int(height)))

            frame, balls = track_balls(frame)

            if not no_colors_tracked:
                c = ["a", "b", "c"]
                for ball in balls:
                    bid = ball['ID']
                    if bid == "d":
                        continue
                    if bid not in balls_locations:
                        balls_locations[bid] = []
                    if bid in c:
                        balls_locations[bid].append(ball['centroid'])
                        c.remove(bid)
                    else:
                        balls_locations[bid].pop()
                        balls_locations[bid].append([0,0])
                for i in c:
                    balls_locations[i].append([0,0])

                # WIP
                for k, v in balls_locations.items():
                    if k == "a":
                        if v[-1] != [0,0]:
                            print("yey")
                            if len(v) > 1:
                                if v[-2] == [0,0]:
                                    newest_coordinates = v.pop()
                                    last_coordinates, number_of_additions = get_last_coordinates_and_number_of_emptys(v)
                                    v = v[:-number_of_additions]
                                    distance = np.linalg.norm(np.array(newest_coordinates) - np.array(last_coordinates), axis=0)
                                    one_step = distance/number_of_additions
                                    for s in range(1,number_of_additions):
                                        v.append(last_coordinates + s*one_step)
                                    v.append(newest_coordinates)
                        else: print("noooooooooo")

            if not no_colors_tracked:
                for ball in balls:
                    balls_last_known_locations[ball['ID']] = ball['centroid']

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Video", frame)

            # temporary
            # with open("test.csv", "a", newline='') as fp:
            #     writer = csv.writer(fp)
            #     for b in balls:
            #             l = [video_name, current_frame_counter, b["centroid"], b["ID"]]
            #             writer.writerow(l)
            # current_frame_counter += 1

            if cv2.waitKey(1) & 0xFF == 27:
                break
    print(balls_locations["a"])

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
