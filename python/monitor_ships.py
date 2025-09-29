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

centers, bboxes, cleaned_mask = ip.detect_ship_centers_from_mask(pred_mask, frame.shape)

# for (cx, cy) in centers:
#     cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

# cv2.imshow("Ships", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
# print(centers)

# ---------- OIL SPILL CENTER ----------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    (pred_mask == 3).astype(np.uint8), connectivity=8
)

# Skip label 0 (background)
oil_centers = []
for i in range(1, num_labels):
    cx, cy = centroids[i]  # (x, y) center of mass
    oil_centers.append((int(cx), int(cy)))

# print(f"Detected {len(oil_centers)} oils")
# print("Centers:", oil_centers)

# for (cx, cy) in oil_centers:
#     cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

# cv2.imshow("Oil", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ---------- AIS ----------

# nmea = "!AIVDM,1,1,,A,14eG;o@034o8sd<L9i:a;WF>062D,0*7D"
# if AIS.checksum(nmea):
#     msg = AIS.decode(nmea)
#     print(f"Longitude: {msg.lon}\nLatitude: {msg.lat}")

# for _ in range(10):
#     nmea = AIS.simulate_ais(custom_coords=True)
#     if AIS.checksum(nmea[0]):
#         msg = AIS.decode(nmea[0])
#         print(f"Longitude: {msg.lon}\nLatitude: {msg.lat}")

ships = []

for (cx, cy) in centers:
    nmea = AIS.simulate_ais(custom_coords=True, pixel_coord=(cx, cy))
    if AIS.checksum(nmea[0]):
        msg = AIS.decode(nmea[0])
        # print(f"Ship at pixel ({cx},{cy}) -> Lon: {msg.lon:.6f}, Lat: {msg.lat:.6f}")

        ships.append({
            "id": msg.mmsi,
            "lon": msg.lon,
            "lat": msg.lat,
            "lon_px": cx,
            "lat_px": cy
        })

closest_ship = {
    "ship_id": 0,
    "difference": abs(abs(oil_centers[0][0] - ships[0]["lon_px"]) - abs(oil_centers[0][1] - ships[0]["lat_px"]))
}

for ship in ships:
    lon = ship["lon_px"]
    lat = ship["lat_px"]

    difference = abs(abs(oil_centers[0][0] - lon) - abs(oil_centers[0][1] - lat))

    if closest_ship["difference"] > difference:
        closest_ship["ship_id"] = ship["id"]
        closest_ship["difference"] = difference

ship = next(s for s in ships if s["id"] == closest_ship["ship_id"])
print(f"Closest ship to oil -> Lon: {ship['lon_px']:.6f}, Lat: {ship['lat_px']:.6f}\nOil was spilled by ship with mmsi: {ship['id']}")