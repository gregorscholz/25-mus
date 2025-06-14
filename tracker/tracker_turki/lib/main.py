import os

from ultralytics import YOLO

from track import track

pose_model = YOLO("yolo11n-pose.pt")

for video_name in os.listdir("videos"):
    if not str(video_name).endswith(".mp4"):
        continue

    track(f"videos/{video_name}", pose_model)
