import cv2
import numpy as np
import os
import shutil

# Helper function to paint colors with a specific margin
def paint_colors(frame, color_mask, color_list, paint_color, margin):
    for color in color_list:
        bgr = color[::-1]  # convert RGB to BGR
        lower = np.clip(bgr - margin, 0, 255)
        upper = np.clip(bgr + margin, 0, 255)
        mask = cv2.inRange(frame, lower, upper)
        color_mask[mask > 0] = paint_color

# Load video
cap = cv2.VideoCapture("video/ManchaSat_compressed.mp4")
if not cap.isOpened():
    raise ValueError("Error opening video file")
else:
    if os.path.isdir("./mask_frames"):
        shutil.rmtree("./mask_frames")

# Define color lists (RGB)
ship_colors = [
    np.array([124, 19, 123]), # purple
    np.array([0, 126, 0]),    # green
    np.array([250, 22, 0]),   # red
    np.array([163, 40, 41]),  # dark red
    np.array([250, 238, 0]),  # yellow
    np.array([252, 163, 0]),  # orange
]

sargassum_colors = [
    np.array([108, 103, 108])
]

oil_colors = [
    np.array([0, 0, 0])
]

# Margins
ship_margin = 50
sargassum_margin = 20
oil_margin = 25

# Create output directory
output_dir = "mask_frames"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Initialize blank color mask
    color_mask = np.zeros_like(frame, dtype=np.uint8)

    # Paint each category with specific margins
    paint_colors(frame, color_mask, sargassum_colors, [128, 128, 128], sargassum_margin)    # gray
    paint_colors(frame, color_mask, ship_colors, [0, 0, 255], ship_margin)                  # red
    paint_colors(frame, color_mask, oil_colors, [255, 255, 255], oil_margin)                # white

    # Save the colored mask
    mask_filename = os.path.join(output_dir, f"mask_{frame_count:04d}.png")
    cv2.imwrite(mask_filename, color_mask)

    frame_count += 1

cap.release()
print(f"Done! Saved {frame_count} mask frames.")
