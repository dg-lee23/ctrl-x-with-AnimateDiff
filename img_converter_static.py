# given an image,
# convert it to a 16-frame, "static" video for DM input 

import cv2
import numpy as np

image_path = "./assets/images/<your_image>.jpg"
video_path = "./assets/videos/<your_video>.mp4"

image = cv2.imread(image_path)
image = cv2.resize(image, (512, 512))

total_frames = 16
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 8
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (512, 512))

for _ in range(total_frames):
    video_writer.write(image)

video_writer.release()

print(f"Static video saved to {video_path}")
