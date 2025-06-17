import numpy as np


class PoseDetector:
    def __init__(self, model):
        self.model = model
        self.keypoints = []

    def track(self, frame):
        result = self.model.predict(frame, verbose=False)[0]
        frame, pose = result.plot(boxes=False), result.keypoints.xy[0]

        filtered_keypoints = {
            "left_shoulder": pose[5].tolist(),
            "right_shoulder": pose[6].tolist(),
            "left_elbow": pose[7].tolist(),
            "right_elbow": pose[8].tolist(),
            "left_hand": pose[9].tolist(),
            "right_hand": pose[10].tolist(),
        }

        self.keypoints.append(filtered_keypoints)

        return frame

    def angle(self, shoulder, elbow, wrist):
        a = np.array(shoulder)
        b = np.array(elbow)
        c = np.array(wrist)

        vec1 = a - b
        vec2 = c - b

        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg
