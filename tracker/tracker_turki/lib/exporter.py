import csv
import os

CSV_FILE_PATH ="out.csv"

def export_to_csv(video_name, last_frame, first_frame, balls_location, pose_keypoints):

    contains_header = False
    if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) != 0:
            contains_header = True


    with open(CSV_FILE_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not contains_header:
            writer.writerow(["video", "balls", "hands", "elbows", "shoulders"])

        balls_a = []
        balls_b = []
        balls_c = []

        left_hands = []
        right_hands = []

        left_elbows = []
        right_elbows = []

        left_shoulders = []
        right_shoulders = []

        for i in range(first_frame, last_frame):
            balls_a.append(balls_location["a"][i])
            balls_b.append(balls_location["b"][i])
            balls_c.append(balls_location["c"][i])

            left_hands.append(pose_keypoints[i]["left_hand"])
            right_hands.append(pose_keypoints[i]["right_hand"])

            left_elbows.append(pose_keypoints[i]["left_elbow"])
            right_elbows.append(pose_keypoints[i]["right_elbow"])

            left_shoulders.append(pose_keypoints[i]["left_shoulder"])
            right_shoulders.append(pose_keypoints[i]["right_shoulder"])

        balls = [balls_a, balls_b, balls_c]
        hands = [left_hands, right_hands]
        elbows = [left_elbows, right_elbows]
        shoulders = [left_shoulders, right_shoulders]

        writer.writerow([video_name, balls, hands, elbows, shoulders])
