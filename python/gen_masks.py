import cv2
import numpy as np
import os
import shutil

# Helper function to paint colors with a specific margin
def paint_colors(frame, color_mask, color_list, paint_color, margin, type, cutoffs=None):
    for color in color_list:
        bgr = color[::-1]  # convert RGB to BGR
        match type:
            case "ship":
                upper = np.clip(bgr + margin, 0, 255)
                lower = np.clip(bgr - int(margin * 1.5), 0, 255)
            case _:
                upper = np.clip(bgr + margin, 0, 255)
                lower = np.clip(bgr - margin, 0, 255)

        mask = cv2.inRange(frame, lower, upper)

        # Apply cutoffs (top, bottom, left, right)
        if cutoffs is not None:
            top, bottom, left, right = cutoffs
            h, w = mask.shape[:2]
            # Mask everything outside the allowed region
            mask[:top, :] = 0
            mask[bottom:, :] = 0
            mask[:, :left] = 0
            mask[:, right:] = 0

        color_mask[mask > 0] = paint_color


# Load video
cap = cv2.VideoCapture("video/ManchaSat.mp4")
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

oil_colors = [
    np.array([0, 0, 0])
]

# Margins
ship_margin = 50
oil_margin = 40

# Create output directory
output_dir = "mask_frames"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

# Define cutoffs: (top, bottom, left, right)
# Example: exclude 200 px from top and bottom, and 300 px from left and right
cutoffs = (19, 850, 77, 1875)  # adjust according to your video resolution

while True:
    ret, frame = cap.read()
    if not ret:
        break

    color_mask = np.zeros_like(frame, dtype=np.uint8)

    # Paint masks only in the central region
    paint_colors(frame, color_mask, ship_colors, [0, 0, 255], ship_margin, "ship", cutoffs)
    paint_colors(frame, color_mask, oil_colors, [255, 255, 255], oil_margin, "oil", cutoffs)

    mask_filename = os.path.join(output_dir, f"mask_{frame_count:04d}.png")
    cv2.imwrite(mask_filename, color_mask)

    frame_count += 1

cap.release()
print(f"Done! Saved {frame_count} mask frames (masks only in central region).")
