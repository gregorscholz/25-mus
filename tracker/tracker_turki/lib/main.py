import os

from ultralytics import YOLO

from track import track

pose_model = YOLO("yolo11n-pose.pt")

# single video
# track(f"videos/ia1-1.mp4", pose_model)

error_videos = []

# all videos
for video_name in os.listdir("videos"):
    if not str(video_name).endswith(".mp4"):
        continue

    try:
        track(f"videos/{video_name}", pose_model)
    except Exception as e:
        print(f"Error: {e} in video {video_name}")
        error_videos.append(video_name)

print(error_videos)
