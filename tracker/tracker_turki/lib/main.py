import os

from track import track

for video_name in os.listdir("videos"):
    if not str(video_name).endswith(".mp4"):
        continue

    track(f"videos/{video_name}")