def format_lists(last_frame, first_frame, ball_location, pose_keypoints):
    ball_a = []
    ball_b = []
    ball_c = []

    left_hands = []
    right_hands = []

    left_elbows = []
    right_elbows = []

    left_shoulders = []
    right_shoulders = []

    for i in range(first_frame, last_frame + 1):
        ball_a.append(ball_location["a"][i])
        ball_b.append(ball_location["b"][i])
        ball_c.append(ball_location["c"][i])

        left_hands.append(pose_keypoints[i]["left_hand"])
        right_hands.append(pose_keypoints[i]["right_hand"])

        left_elbows.append(pose_keypoints[i]["left_elbow"])
        right_elbows.append(pose_keypoints[i]["right_elbow"])

        left_shoulders.append(pose_keypoints[i]["left_shoulder"])
        right_shoulders.append(pose_keypoints[i]["right_shoulder"])

    return {
        "ball_a": ball_a,
        "ball_b": ball_b,
        "ball_c": ball_c,
        "left_hand": left_hands,
        "right_hand": right_hands,
        "left_elbow": left_elbows,
        "right_elbow": right_elbows,
        "left_shoulder": left_shoulders,
        "right_shoulder": right_shoulders,
    }


def normalize(coords, index):
    if index >= 2:
        raise Exception("Index out of range")

    all_lists = [
        coords["ball_a"],
        coords["ball_b"],
        coords["ball_c"],
        coords["left_hand"],
        coords["right_hand"],
        coords["left_elbow"],
        coords["right_elbow"],
        coords["left_shoulder"],
        coords["right_shoulder"],
    ]
    all_index = [point[index] for lst in all_lists for point in lst]

    max_index = max(all_index)
    min_index = min(all_index)

    normalized_lists = []

    for lst in all_lists:
        normalized_lst = []
        if index == 0:
            normalized_lst = [
                [(x - min_index) / (max_index - min_index), y] for x, y in lst
            ]
        elif index == 1:
            normalized_lst = [
                [x, (y - min_index) / (max_index - min_index)] for x, y in lst
            ]

        normalized_lists.append(normalized_lst)

    return {
        "ball_a": normalized_lists[0],
        "ball_b": normalized_lists[1],
        "ball_c": normalized_lists[2],
        "left_hand": normalized_lists[3],
        "right_hand": normalized_lists[4],
        "left_elbow": normalized_lists[5],
        "right_elbow": normalized_lists[6],
        "left_shoulder": normalized_lists[7],
        "right_shoulder": normalized_lists[8],
    }
