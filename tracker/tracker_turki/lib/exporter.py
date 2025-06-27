import csv
import os

CSV_FILE_PATH = "out.csv"


def export_to_csv(video_name, coords):
    contains_header = False
    if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) != 0:
        contains_header = True

    with open(CSV_FILE_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not contains_header:
            writer.writerow(["video", "ball_a", "ball_b", "ball_c", "hand_left", "hand_right", "elbow_left", "elbow_right", "shoulder_left", "shoulder_right"])

        ball_a = coords["ball_a"]
        ball_b = coords["ball_b"]
        ball_c = coords["ball_c"]
        hand_left = coords["left_hand"]
        hand_right = coords["right_hand"]
        elbow_left = coords["left_elbow"]
        elbow_right = coords["right_elbow"]
        shoulder_left = coords["left_shoulder"]
        shoulder_right = coords["right_shoulder"]

        writer.writerow([video_name, ball_a, ball_b, ball_c, hand_left, hand_right, elbow_left, elbow_right, shoulder_left, shoulder_right])
