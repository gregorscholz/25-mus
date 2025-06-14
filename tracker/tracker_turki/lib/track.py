import string

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from exporter import export_to_csv
from pose import PoseDetector

RADIUS_COLOR = 3
THRESHOLD_COLOR = 60

ball_colors = {}
no_colors_tracked = True
balls_locations = {}


def _find_closest_previous_ball_and_update_ball_colors(frame, ball):
    # find the closest ball

    # temporary
    avg_color = _get_avg_ball_color(frame, ball)

    if avg_color is None:
        return ball

    ck = "d"
    minCDistance = 100000

    # check for fitting color
    for k, v in ball_colors.items():
        cDistance = np.linalg.norm(avg_color - v)
        if cDistance < minCDistance:
            minCDistance = cDistance
            ck = k

    dk = "d"
    minDistance = 100000
    # check for shortest distance
    for k, v in _get_last_known_locations().items():
        distance = np.linalg.norm(np.array(v) - np.array(ball["centroid"]), axis=0)
        if distance < minDistance:
            minDistance = distance
            dk = k

    # check distance supports fitting color
    if dk != ck:
        ball["ID"] = "d"
        return ball
    else:
        ball["ID"] = dk

    # update ball_colors
    weights = [0.95, 0.05]
    bid = ball["ID"]

    ball_colors[bid] = np.average(
        [ball_colors[bid], avg_color], weights=weights, axis=0
    )

    return ball


def _check_balls(frame, balls):
    for ball in balls:
        if ball["ID"] == "d":
            ball = _find_closest_previous_ball_and_update_ball_colors(frame, ball)

    counter = _count_tracked_ball_ids(balls)

    # multiple balls of one color
    for key, value in counter.items():
        if key == "d":
            continue
        if value > 1:
            for ball in balls:
                if ball["ID"] == key:
                    ball = _find_closest_previous_ball_and_update_ball_colors(
                        frame, ball
                    )

    return balls


def _get_avg_ball_color(frame, ball):
    colors = []
    x = ball["centroid"][0]
    y = ball["centroid"][1]

    for dy in range(-RADIUS_COLOR, RADIUS_COLOR + 1):
        for dx in range(-RADIUS_COLOR, RADIUS_COLOR + 1):
            nx, ny = x + dx, y + dy

            if 0 <= nx < frame.shape[1] and 0 <= ny < frame.shape[0]:
                color = frame[int(ny), int(nx)]
                colors.append(color)

    if colors:
        return np.mean(colors, axis=0)  # avg_color
    else:
        return None


def _identify_balls(frame, balls):
    global ball_colors
    global no_colors_tracked

    if no_colors_tracked:
        # check for 3 balls in frame
        if len(balls) != 3:
            return balls

    for i, ball in enumerate(balls):
        avg_color = _get_avg_ball_color(frame, ball)

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


def _draw_bbox(image, balls):
    image_h, image_w = image.shape[:2]

    for ball in balls:
        recColor = (0, 0, 0)
        match ball["ID"]:
            case "a":
                recColor = (255, 0, 0)
            case "b":
                recColor = (0, 255, 0)
            case "c":
                recColor = (0, 0, 255)

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


def _track(frame, bbox):
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


def _track_balls(frame):
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

        balls = _track(frame, pred_bbox)
        balls = _identify_balls(frame, balls)
        balls = _check_balls(frame, balls)
        frame_with_boxes = _draw_bbox(frame, balls)

        if frame_with_boxes is None:
            return frame

        return frame_with_boxes, balls
    return None


def _get_last_known_locations() -> dict:
    locations = {}
    for k, v in balls_locations.items():
        for location in reversed(v):
            if location != [0, 0]:
                locations[k] = location
                break
    return locations


def _interpolate_missing_locations():
    # WIP
    for k, v in balls_locations.items():
        print(v[-2:])
        if v[-1] != [0, 0]:
            if len(v) > 1:
                if v[-2] == [0, 0]:
                    newest_location = v.pop()
                    counter = 0
                    last_location = [0, 0]
                    for location in reversed(v):
                        if location != [0, 0]:
                            last_location = location
                            break
                        else:
                            counter += 1

                    # check if last location was found
                    if last_location == [0, 0]:
                        # no interpolation possible
                        # add newest location again
                        v.append(newest_location)
                        break

                    if k == "a":
                        print(counter)
                        print(v[-(counter + 1) :])
                    # for i in range(0,counter):
                    #     temp = v.pop()
                    #     if k == "a":
                    #         print(f"removed: {temp}")
                    v = v[:-counter]
                    if k == "a":
                        print("danach")
                        print(v[-(counter + 1) :])

                    # distance = np.linalg.norm(
                    #     np.array(newest_coordinates) - np.array(last_coordinates),
                    #     axis=0,
                    # )
                    # one_step = distance / number_of_additions
                    # for s in range(1, number_of_additions):
                    #     v.append(last_coordinates + s * one_step)

                    # add newest location again
                    v.append(newest_location)
        print(v[-2:])


def _save_locations(balls):
    c = ["a", "b", "c"]
    if not no_colors_tracked:
        for ball in balls:
            bid = ball["ID"]
            if bid == "d":
                continue
            if bid not in balls_locations:
                balls_locations[bid] = []
            if bid in c:
                balls_locations[bid].append(ball["centroid"])
                c.remove(bid)
            else:
                balls_locations[bid].pop()
                balls_locations[bid].append([0, 0])

    # add [0,0] for balls not tracked in that frame
    for i in c:
        if i not in balls_locations:
            balls_locations[i] = []
        balls_locations[i].append([0, 0])

    # _interpolate_missing_locations()


def _count_tracked_ball_ids(balls) -> dict:
    counter = {"a": 0, "b": 0, "c": 0, "d": 0}
    for ball in balls:
        match ball["ID"]:
            case "a":
                counter["a"] += 1
            case "b":
                counter["b"] += 1
            case "c":
                counter["c"] += 1
            case _:
                counter["d"] += 1
    return counter


saved_model_loaded_ball = tf.saved_model.load(
    "ball_weights", tags=[tag_constants.SERVING]
)
infer_ball = saved_model_loaded_ball.signatures["serving_default"]  # type: ignore


def track(video_name, pose_model):
    pose_detector = PoseDetector(pose_model)

    cap = cv2.VideoCapture(video_name)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    global balls_locations
    current_frame_counter = 0
    global no_colors_tracked

    balls_locations = {}
    current_frame_counter = 1
    no_colors_tracked = True

    last_frame = 0
    first_frame = 0
    first_frame_found = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video processing complete")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (int(width), int(height)))

        # Returns a frame with the pose
        _ = pose_detector.track(frame)

        # optional preprocessing
        # frame_filtered = filter_balls(frame)

        frame, balls = _track_balls(frame)

        _save_locations(balls)

        if not no_colors_tracked:
            # check for multiple balls of one color
            counter = _count_tracked_ball_ids(balls)

            ball_counter = 0
            for key, value in counter.items():
                if value == 1:
                    ball_counter += 1

            if ball_counter == 3:
                last_frame = current_frame_counter

            if ball_counter == 3 and not first_frame_found:
                first_frame_found = True
                first_frame = current_frame_counter

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        current_frame_counter += 1

    export_to_csv(
        video_name, last_frame, first_frame, balls_locations, pose_detector.keypoints
    )

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print(balls_locations["a"])
