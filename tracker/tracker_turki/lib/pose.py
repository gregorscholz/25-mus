import numpy as np


class PoseDetector:
    def __init__(self, model):
        self.model = model
        self.keypoints = []

    def track(self, frame):
        result = self.model.predict(frame)[0]
        frame, pose = result.plot(boxes=False), result.keypoints.xy[0]

        filtered_keypoints = {
            "leftShoulder": pose[5].tolist(),
            "rightShoulder": pose[6].tolist(),
            "leftElbow": pose[7].tolist(),
            "rightElbow": pose[8].tolist(),
            "leftHand": pose[9].tolist(),
            "rightHand": pose[10].tolist(),
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
