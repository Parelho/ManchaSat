import cv2
import numpy as np
import tensorflow as tf
from image_processing import ImageProcessing as ip

model = tf.keras.models.load_model("jupyter/segmentation_unet.h5", compile=False)

img_path = "grayscale_frames/frame_0900.png"
frame = cv2.imread(img_path)
if frame is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

input_tensor = ip.preprocess_frame(frame)
pred = model.predict(input_tensor, verbose=0)
pred_mask = ip.decode_mask(pred, frame.shape)

oil_pixels = np.sum(pred_mask == 3)
print(f"Oil spill area: {oil_pixels} pixels")

total_pixels = pred_mask.size
percentage = (oil_pixels / total_pixels) * 100
print(f"Oil spill covers {percentage:.2f}% of the image")
