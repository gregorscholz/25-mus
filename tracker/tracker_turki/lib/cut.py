import moviepy

def trim_left_edge(input_video_path, output_video_path, trim_width):
    video = moviepy.VideoFileClip(input_video_path)
    trimmed_video = video.cropped(x1=trim_width, y1=0, x2=video.w, y2=video.h)
    trimmed_video.write_videofile(output_video_path, codec='libx264')

input_video = "videos/ic11-0.mp4"
output_video = "videos/0ic11-0.mp4"
trim_width = 100

trim_left_edge(input_video, output_video, trim_width)
