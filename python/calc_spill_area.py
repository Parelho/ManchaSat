import cv2
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # remove tensorflow warnings
import tensorflow as tf
from image_processing import ImageProcessing as ip
from resolution import Resolution as rs

model = tf.keras.models.load_model("jupyter/segmentation_unet.h5", compile=False)

# ---------- IMAGE AREA ----------

img_path = "grayscale_frames/frame_1200.png"
frame = cv2.imread(img_path)
if frame is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

input_tensor = ip.preprocess_frame(frame)
pred = model.predict(input_tensor, verbose=0)
pred_mask = ip.decode_mask(pred, frame.shape)

oil_pixels = np.sum(pred_mask == 3)
total_pixels = pred_mask.size
percentage = (oil_pixels / total_pixels)

# ---------- SPILL AREA ----------

distance = 430 # in km
fov_horizontal = 65
fov_vertical = 48

horizontal_resolution, vertical_resolution = rs.get_resolution(distance, fov_horizontal, fov_vertical)
horizontal_resolution = horizontal_resolution / 256 # km/px
vertical_resolution = vertical_resolution / 256 # km/px

print(f"horizontal resolution: {horizontal_resolution:.2f} km/px\nvertical resolution: {vertical_resolution:.2f} km/px")

pixel_area = horizontal_resolution * vertical_resolution # km^2 per pixel

spill_area = oil_pixels * pixel_area # km^2 of oil spill

print(f"Oil spill area: {spill_area:.2f} km^2 ({percentage*100:.2f}%)")
