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
            writer.writerow(["video", "balls", "hands", "elbows", "shoulders"])

        balls = [coords["ball_a"], coords["ball_b"], coords["ball_c"]]
        hands = [coords["left_hand"], coords["right_hand"]]
        elbows = [coords["left_elbow"], coords["right_elbow"]]
        shoulders = [coords["left_shoulder"], coords["right_shoulder"]]

        writer.writerow([video_name, balls, hands, elbows, shoulders])
