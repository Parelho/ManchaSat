import cv2
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # remove tensorflow warnings
import tensorflow as tf
from image_processing import ImageProcessing as ip
from ais import AIS
import glob

model = tf.keras.models.load_model("jupyter/segmentation_unet.h5", compile=False)

# ---------- GLOBAL PREVIOUS SPILL MASK ----------
prev_oil_mask = None  # store mask of previous frame

def get_new_spill_mask(pred_mask):
    global prev_oil_mask
    current_mask = (pred_mask == 3).astype(np.uint8)

    if prev_oil_mask is None:
        new_spill = current_mask
    else:
        new_spill = np.logical_and(current_mask, prev_oil_mask == 0).astype(np.uint8)

    prev_oil_mask = current_mask.copy()
    return new_spill

def get_closest_ship(image):
    # ---------- IMAGE ----------
    img_path = image
    frame = cv2.imread(img_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    input_tensor = ip.preprocess_frame(frame)
    pred = model.predict(input_tensor, verbose=0)
    pred_mask = ip.decode_mask(pred, frame.shape)

    centers, bboxes, cleaned_mask = ip.detect_ship_centers_from_mask(pred_mask, frame.shape)

    # ---------- NEW OIL SPILL REGION ----------
    new_spill_mask = get_new_spill_mask(pred_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(new_spill_mask, connectivity=8)

    oil_centers = [(int(cx), int(cy)) for cx, cy in centroids[1:]]

    if oil_centers:
        ships = []
        for (cx, cy) in centers:
            nmea = AIS.simulate_ais(custom_coords=True, pixel_coord=(cx, cy))
            if AIS.checksum(nmea[0]):
                msg = AIS.decode(nmea[0])
                ships.append({
                    "mmsi": msg.mmsi,
                    "lon": msg.lon,
                    "lat": msg.lat,
                    "lon_px": cx,
                    "lat_px": cy
                })

        closest_ship = None
        if ships:
            closest_ship = ships[0]
            min_diff = abs(oil_centers[0][0] - ships[0]["lon_px"]) + abs(oil_centers[0][1] - ships[0]["lat_px"])

            for ship in ships[1:]:
                diff = abs(oil_centers[0][0] - ship["lon_px"]) + abs(oil_centers[0][1] - ship["lat_px"])
                if diff < min_diff:
                    closest_ship = ship
                    min_diff = diff

        return closest_ship
    else:
        return None

# ---------- MAIN LOOP ----------
images_dir = "grayscale_frames"
ships_near_oil = []

for file_path in sorted(glob.glob(os.path.join(images_dir, "*.png"))):
    closest_ship = get_closest_ship(file_path)

    if closest_ship is not None:
        existing_ship = next((ship for ship in ships_near_oil if ship["mmsi"] == closest_ship["mmsi"]), None)
        if existing_ship is None:
            closest_ship["proximity_count"] = 1
            ships_near_oil.append(closest_ship)
            print(f"New ship near oil: {closest_ship}")
        else:
            existing_ship["proximity_count"] += 1
            print(f"Ship {closest_ship['mmsi']} has the highest proximity count at: {existing_ship['proximity_count']}")

print(ships_near_oil)
