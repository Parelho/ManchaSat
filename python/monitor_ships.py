import cv2
import numpy as np
import os
from tflite_runtime.interpreter import Interpreter
from image_processing import ImageProcessing as ip
from calc_spill_area import CalcSpillArea
from ais import AIS
import glob
import random
import time
import csv
import traceback
import sys

model_path = "model.tflite"
interpreter = Interpreter(model_path=model_path, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
all_ships = [] # Used to track ships that stopped sending AIS

analyzer = CalcSpillArea()

oil_present = False

# ---------- SHIP–OIL OVERLAP CHECK ----------
def check_overlap(oil_mask, bbox):
    x, y, w, h = bbox
    ship_mask = np.zeros_like(oil_mask, dtype=np.uint8)
    cv2.rectangle(ship_mask, (x, y), (x + w, y + h), 255, -1)
    overlap = cv2.bitwise_and(oil_mask, ship_mask)
    return np.count_nonzero(overlap) > 0

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

prev_closest_ship = None
def calculate_closest_ship(ships, oil_centers, prev_ship=None, inertia_weight=0.7):
    closest_ship = ships[0]
    min_diff = np.hypot(oil_centers[0][0] - closest_ship["lon_px"], oil_centers[0][1] - closest_ship["lat_px"])
    # print(f"min_diff_og: {min_diff}, mmsi: {closest_ship['mmsi']}")
    for ship in ships[1:]:
        diff = np.hypot(oil_centers[0][0] - ship["lon_px"], oil_centers[0][1] - ship["lat_px"])
        # print(f"diff: {diff}, mmsi: {ship['mmsi']}")
        # print(ship)
        if diff < min_diff:
            closest_ship = ship
            min_diff = diff

    if prev_ship is not None:
        # Penalize switching to a new ship unless it's significantly closer
        prev_diff = np.hypot(oil_centers[0][0] - prev_ship["lon_px"], oil_centers[0][1] - prev_ship["lat_px"])
        if prev_diff * inertia_weight < min_diff:
            closest_ship = prev_ship
            min_diff = prev_diff

    return closest_ship, min_diff

def get_closest_ship(image, ships_near_oil):
    # ---------- IMAGE ----------
    frame = cv2.imread(image)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image}")

    input_tensor = ip.preprocess_frame(frame)

    # Use the TFLite interpreter instead of model.predict
    input_index = input_details[0]['index']
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_index = output_details[0]['index']
    pred = interpreter.get_tensor(output_index)

    pred_mask = ip.decode_mask(pred, frame.shape)
    centers, bboxes, cleaned_mask = ip.detect_ship_centers_from_mask(pred_mask, frame.shape)

    # ---------- NEW OIL SPILL REGION ----------
    global old_oil_spill
    new_spill_mask = get_new_spill_mask(pred_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(new_spill_mask, connectivity=8)

    oil_centers = [(int(cx), int(cy)) for cx, cy in centroids[1:]]

    if np.count_nonzero(new_spill_mask) > 0:
        num_labels_new, _, stats_new, centroids_new = cv2.connectedComponentsWithStats(new_spill_mask, connectivity=8)
        if len(centroids_new) > 1:
            # Use centroid of the largest new oil region
            largest_idx = np.argmax(stats_new[1:, cv2.CC_STAT_AREA]) + 1
            new_oil_center = (int(centroids_new[largest_idx][0]), int(centroids_new[largest_idx][1]))
            oil_present = True
        else:
            new_oil_center = None
            oil_present = False
    else:
        new_oil_center = None
        oil_present = False

    if centers:
        csv_reader = []
        rows = []
        with open("input.csv", mode="r", newline="") as f:
            csv_reader = csv.reader(f)
            next(csv_reader, None)  # skip header
            rows = list(csv_reader)  # store all rows in memory

        ships = []
        for row in rows:
            ship = {
                "mmsi": row[0],
                "lon": float(row[1]),
                "lat": float(row[2]),
                "lon_px": 0,
                "lat_px": 0
            }
            ships.append(ship)

        ships = AIS.lat_lon_to_px(centers, ships)

        for new_ship in ships:
            existing_ship = next((ship for ship in all_ships if ship["mmsi"] == new_ship["mmsi"]), None)
            if existing_ship is None:
                all_ships.append(ship)
                # print("appended to all ships")
        # print(ships)
        # print(oil_centers)

        closest_ship = None
        closest_ship_difference = 9999999
        spiller_is_hidden = False

        for ship in ships:
            difference_spill_lon = abs(oil_centers[0][0] - ship["lon_px"])
            difference_spill_lat = abs(oil_centers[0][1] - ship["lat_px"])
            difference = difference_spill_lon + difference_spill_lat

            if (difference < closest_ship_difference):
                closest_ship = ship
                closest_ship_difference = difference

        oil_pixels = np.sum(pred_mask == 3)

        result = analyzer.analyze(frame, oil_pixels)

        # oil_mask_vis = (pred_mask == 1).astype(np.uint8) * 255
        # cv2.imwrite("oil_mask_latest.jpg", oil_mask_vis)
        # ---------- DEBUG VISUALIZATION: SHIPS + OIL MASK ----------
        debug_img = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)

        # Create a red overlay for the oil mask (class 3)
        oil_mask_color = np.zeros_like(debug_img)
        oil_mask_color[pred_mask == 3] = (0, 0, 255)
        debug_img = cv2.addWeighted(debug_img, 1.0, oil_mask_color, 0.4, 0)

        # Draw bounding boxes and MMSI labels for all ships
        for i, ship in enumerate(ships):
            if i < len(bboxes):
                x, y, w, h = bboxes[i]
                mmsi_text = f"mmsi: {ship.get('mmsi', '?')}"
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(debug_img, mmsi_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw bounding box for the oil region (class 3)
        oil_mask = (pred_mask == 3).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(oil_mask, connectivity=8)

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(debug_img, "", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Upscale for clarity
        debug_img = cv2.resize(debug_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Save debug image
        cv2.imwrite("debug_ships.png", debug_img)

        return closest_ship, spiller_is_hidden, result
    else:
        return None, False, 0

# ---------- MAIN LOOP ----------
ships_near_oil = []
# img_count = 200

for i in range(1):
    try:
        # closest_ship, spiller_is_hidden, oil_pixels = get_closest_ship(f"grayscale_frames/frame_0{img_count}.png", ships_near_oil)
        closest_ship, spiller_is_hidden, oil_pixels = get_closest_ship(f"image.jpg", ships_near_oil)
        # img_count += 1
        output = ""

        if closest_ship is not None:
            # print(len(closest_ship))
            print(f"Oil pixels: {oil_pixels}")

            existing_ship = next((ship for ship in ships_near_oil if ship["mmsi"] == closest_ship["mmsi"]), None)
            if existing_ship is None:
                closest_ship["proximity_count"] = 1
                ships_near_oil.append(closest_ship)
                print(f"New ship near oil: {closest_ship}")
            else:
                existing_ship["proximity_count"] += 1
                # print("The spilling ship is hiding it's AIS signal\n")

            top_ship = max(ships_near_oil, key=lambda s: s["proximity_count"])
            output = f"A{int(oil_pixels['estimated_area_km2'])},mmsi{top_ship['mmsi']}"
            with open("output.txt", "w") as f:
                    f.write(output)
        elif not oil_present:
            output = "A0,mmsi0"
            with open("output.txt", "w") as f:
                    f.write(output)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        # if ships_near_oil:
        #     top_ship = max(ships_near_oil, key=lambda s: s["proximity_count"])
        #     print(f"Ship with highest proximity count: {top_ship['mmsi']} ({top_ship['proximity_count']}), at lon: {top_ship['lon']} lat: {top_ship['lat']}")

        # print(ships_near_oil)