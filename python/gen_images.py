import cv2
import os
import shutil

# Load video
cap = cv2.VideoCapture("video/ManchaSat_compressed.mp4")
if not cap.isOpened():
    raise ValueError("Error opening video file")
else:
    if os.path.isdir("./grayscale_frames"):
        shutil.rmtree("./grayscale_frames")

# Create output directory
output_dir = "grayscale_frames"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Save grayscale frame
    frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_filename, gray_frame)

    frame_count += 1

cap.release()
print(f"Done! Saved {frame_count} grayscale frames.")
