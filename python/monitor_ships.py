import cv2
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # remove tensorflow warnings
import tensorflow as tf
from image_processing import ImageProcessing as ip
from ais import AIS

model = tf.keras.models.load_model("jupyter/segmentation_unet.h5", compile=False)

# ---------- IMAGE ----------

img_path = "grayscale_frames/frame_1200.png"
frame = cv2.imread(img_path)
if frame is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

input_tensor = ip.preprocess_frame(frame)
pred = model.predict(input_tensor, verbose=0)
pred_mask = ip.decode_mask(pred, frame.shape)

# ---------- AIS ----------

# checksum
# nmea = "!AIVDM,1,1,,A,14eG;o@034o8sd<L9i:a;WF>062D,0*7D"

# if AIS.checksum(nmea):
#     msg = AIS.decode(nmea)

#     print(f"Longitude: {msg.lon}\nLatitude: {msg.lat}")

for _ in range(10):
    nmea = AIS.simulate_ais()
    if AIS.checksum(nmea[0]):
        msg = AIS.decode(nmea[0])
        print(f"Longitude: {msg.lon}\nLatitude: {msg.lat}")