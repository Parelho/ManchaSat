import cv2
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # remove tensorflow warnings
import tensorflow as tf
from image_processing import ImageProcessing as ip
from ais import AIS
import glob
import random
import time

model = tf.keras.models.load_model("jupyter/model.h5", compile=False)
all_ships = []  # Used to track ships that stopped sending AIS

# ---------- GLOBAL PREVIOUS SPILL MASK ----------
prev_oil_mask = None  # store mask of previous frame
old_oil_spill = None  # store the position where spill first appeared


def get_new_spill_mask(pred_mask):
    global prev_oil_mask
    current_mask = (pred_mask == 3).astype(np.uint8)

    if prev_oil_mask is None:
        new_spill = current_mask
    else:
        new_spill = np.logical_and(current_mask, prev_oil_mask == 0).astype(np.uint8)

    prev_oil_mask = current_mask.copy()
    return new_spill


def calculate_closest_ship(ships, oil_centers):
    closest_ship = ships[0]
    diff = 0
    min_diff = abs(oil_centers[0][0] - ships[0]["lon_px"]) + abs(oil_centers[0][1] - ships[0]["lat_px"])

    for ship in ships[1:]:
        diff = abs(oil_centers[0][0] - ship["lon_px"]) + abs(oil_centers[0][1] - ship["lat_px"])
        if diff < min_diff:
            closest_ship = ship
            min_diff = diff

    return closest_ship, min_diff


def get_closest_ship_from_data(frame, pred_mask, ships_near_oil):
    global old_oil_spill, all_ships

    centers, bboxes, cleaned_mask = ip.detect_ship_centers_from_mask(pred_mask, frame.shape)

    # ---------- NEW OIL SPILL REGION ----------
    new_spill_mask = get_new_spill_mask(pred_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(new_spill_mask, connectivity=8)
    oil_centers = [(int(cx), int(cy)) for cx, cy in centroids[1:]]

    if oil_centers and old_oil_spill is None:
        old_oil_spill = new_spill_mask

    if centers:
        ships = []
        for (cx, cy) in centers:
            nmea = AIS.simulate_ais(custom_coords=True, pixel_coord=(cx, cy))
            if AIS.checksum(nmea[0]):
                msg = AIS.decode(nmea[0])
                ship = {
                    "mmsi": msg.mmsi,
                    "lon": msg.lon,
                    "lat": msg.lat,
                    "lon_px": cx,
                    "lat_px": cy
                }
                coords_check = AIS.check_coordinates(ship, [0, 0], [256, 256])
                if coords_check is not None:
                    ship["lon"] = coords_check[0]
                    ship["lat"] = coords_check[1]
                    print("Ship faked it's coordinates")
                ships.append(ship)

                # Add new ship to all ships list or update position
                existing_ship = next((s for s in all_ships if s["mmsi"] == msg.mmsi), None)
                if existing_ship is None:
                    all_ships.append(ship)
                else:
                    existing_ship.update(ship)

        closest_ship = None
        spiller_is_hidden = False

        if ships and oil_centers:
            if random.randint(0, 9) < 1:
                closest_ship, _ = calculate_closest_ship(ships, oil_centers)
                ships.remove(closest_ship)  # hides oil spiller

            if len(ships) < len(centers):  # Some ships are hidden
                print("A ship was hidden")

                match_threshold_px = 30
                all_ships_threshold_px = 60

                available_centers = list(centers)
                for ship in list(ships):
                    if not available_centers:
                        break
                    dists = [np.hypot(c[0] - ship["lon_px"], c[1] - ship["lat_px"]) for c in available_centers]
                    min_idx = int(np.argmin(dists))
                    if dists[min_idx] < match_threshold_px:
                        available_centers.pop(min_idx)

                missing_centers = available_centers

                for center in missing_centers:
                    present_mmsi = {s["mmsi"] for s in ships}
                    candidates = [s for s in all_ships if s["mmsi"] not in present_mmsi]

                    if not candidates:
                        print(f"No candidates in all_ships to match missing center {center}")
                        continue

                    dists = [np.hypot(center[0] - c["lon_px"], center[1] - c["lat_px"]) for c in candidates]
                    min_idx = int(np.argmin(dists))
                    best_dist = dists[min_idx]
                    best_ship = candidates[min_idx]

                    if best_dist < all_ships_threshold_px:
                        missing_ship = best_ship.copy()
                        missing_ship["lon_px"] = int(center[0])
                        missing_ship["lat_px"] = int(center[1])
                        coords_check = AIS.check_coordinates(missing_ship, [0, 0], [256, 256])
                        if coords_check is not None:
                            missing_ship["lon"] = coords_check[0]
                            missing_ship["lat"] = coords_check[1]
                        for i, aship in enumerate(all_ships):
                            if aship["mmsi"] == missing_ship["mmsi"]:
                                all_ships[i].update(missing_ship)
                                break
                        ships.append(missing_ship)
                        print(f"Recovered hidden ship {missing_ship['mmsi']} -> center {center} (px dist {best_dist:.1f})")
                    else:
                        print(f"No match in all_ships for center {center} (closest dist {best_dist:.1f})")

            closest_current_ship, current_diff = calculate_closest_ship(ships, oil_centers)
            if old_oil_spill is not None:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(old_oil_spill, connectivity=8)
                if len(centroids) > 1:
                    old_oil_center = (int(centroids[1][0]), int(centroids[1][1]))
                    closest_old_ship, old_diff = calculate_closest_ship(all_ships, [old_oil_center])
                else:
                    closest_old_ship, old_diff = calculate_closest_ship(all_ships, oil_centers)
            else:
                closest_old_ship, old_diff = calculate_closest_ship(all_ships, oil_centers)

            if current_diff < old_diff:
                closest_ship = closest_current_ship
            else:
                closest_ship = closest_old_ship
                spiller_is_hidden = True
                old_oil_spill = new_spill_mask

        return closest_ship, spiller_is_hidden
    else:
        return None, False


# ---------- MAIN LOOP (BATCHED PREDICTION) ----------
start = time.time()
images_dir = "grayscale_frames"
ships_near_oil = []
batch_size = 8  # adjust to fit your GPU memory

# Collect image paths
image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))

# Process in batches
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i : i + batch_size]
    frames = []
    for path in batch_paths:
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {path}")
        frames.append(ip.preprocess_frame(frame))

    # Stack preprocessed frames into a batch
    batch_input = np.concatenate(frames, axis=0)

    # Run a single batched prediction
    preds = model.predict(batch_input, batch_size=batch_size, verbose=0)

    # Decode masks for each frame
    for idx, path in enumerate(batch_paths):
        frame = cv2.imread(path)
        pred_mask = ip.decode_mask(preds[idx:idx+1], frame.shape)
        closest_ship, spiller_is_hidden = get_closest_ship_from_data(frame, pred_mask, ships_near_oil)

        if closest_ship is not None:
            existing_ship = next((ship for ship in ships_near_oil if ship["mmsi"] == closest_ship["mmsi"]), None)
            if existing_ship is None:
                closest_ship["proximity_count"] = 1
                ships_near_oil.append(closest_ship)
                print(f"New ship near oil: {closest_ship}")
            else:
                existing_ship["proximity_count"] += 1

if ships_near_oil:
    top_ship = max(ships_near_oil, key=lambda s: s["proximity_count"])
    print(f"Ship with highest proximity count: {top_ship['mmsi']} ({top_ship['proximity_count']}), "
          f"at lon: {top_ship['lon']} lat: {top_ship['lat']}")

print(ships_near_oil)

elapsed = time.time() - start
print(f"Finished in {elapsed:.2f} seconds")
