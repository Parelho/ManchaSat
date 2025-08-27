import cv2
import os

INPUT_VIDEO = "./video/ManchaSat.mp4"
OUTPUT_VIDEO = "./video/ManchaSat_compressed.mp4"
TARGET_SIZE = (256, 256)
FPS = 30
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise ValueError("Error opening video file")

out = cv2.VideoWriter(OUTPUT_VIDEO, FOURCC, FPS, TARGET_SIZE)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to target size
    frame_resized = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    out.write(frame_resized)
    frame_count += 1

cap.release()
out.release()
print(f"Done! Compressed {frame_count} frames to {OUTPUT_VIDEO}")
