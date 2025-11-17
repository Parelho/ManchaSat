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

def get_closest_ship(image, input, ships_near_oil):
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
        ships = []
        for center in centers:
            ship = {
                "mmsi": 0,
                "lon": 0,
                "lat": 0,
                "lon_px": center[0],
                "lat_px": center[1]
            }
            ships.append(ship)

        csv_reader = []
        rows = []
        with open(input, mode="r", newline="") as f:
            csv_reader = csv.reader(f)
            next(csv_reader, None)  # skip header
            rows = list(csv_reader)  # store all rows in memory

        for row in rows:
            closest_ship_to_row = None
            difference = 999999
            for ship in ships:
                lon_adjusted, lat_adjusted = AIS.lat_lon_to_px(float(row[1]), float(row[2]))

                new_difference = abs(ship["lon_px"] - lon_adjusted) + abs(ship["lat_px"] - lat_adjusted)
                if new_difference < difference:
                    closest_ship_to_row = ship
            
            for ship in ships:
                if ship["lon_px"] == closest_ship_to_row["lon_px"] and ship ["lat_px"] == closest_ship_to_row["lat_px"]:
                    ship["lon"] = float(row[1])
                    ship["lat"] = float(row[2])
        
        # try to update ships with no AIS in current input
        for ship in ships:
            if ship["mmsi"] == 0:
                closest_ship_without_ais = None
                difference = 999999
                for all_ship in all_ships:
                    if all_ship["mmsi"] in ships:
                        continue # skips ship if it already had it's AIS in input
                    else:
                        new_difference = abs(ship["lon_px"] - all_ship["lon_px"]) + abs(ship["lat_px"] - all_ship["lat_px"])
                        if new_difference < difference:
                            closest_ship_without_ais = all_ship
                
                if closest_ship_without_ais is not None:
                    lon_adjusted = (ship["lon_px"] - 976.5) / 5
                    lat_adjusted = (459.5 - ship["lat_px"]) / 4.91
                    ship["mmsi"] = closest_ship_without_ais["mmsi"]
                    ship["lon"] = lon_adjusted
                    ship["lat"] = round(lat_adjusted, 1)

                    for all_ship in all_ships:
                        if closest_ship_without_ais["mmsi"] == all_ship["mmsi"]:
                            all_ship["lon"] = lon_adjusted
                            all_ship["lat"] = round(lat_adjusted, 1)

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

        if not oil_centers:
            return None, False, {"estimated_area_km2": 0}

        spill_x, spill_y = oil_centers[0]

        for ship in ships:
            difference = abs(spill_x - ship["lon_px"]) + abs(spill_y - ship["lat_px"])

            if difference < closest_ship_difference:
                closest_ship = ship
                closest_ship_difference = difference

        oil_pixels = np.sum(pred_mask == 3)

        result = analyzer.analyze(frame, oil_pixels)

        # oil_mask_vis = (pred_mask == 1).astype(np.uint8) * 255
        # cv2.imwrite("oil_mask_latest.jpg", oil_mask_vis)

        return closest_ship, spiller_is_hidden, result
    else:
        return None, False, 0

# ---------- MAIN LOOP ----------
ships_near_oil = []
# img_count = 200

images = sorted(glob.glob("grayscale_frames/*.png"))
inputs = sorted(glob.glob("inputs/*.csv"))

for image_index, image in enumerate(images):
    if image_index % 20 != 0:
        continue
    try:
        ais_rows = AIS.load_ais_csv(inputs[image_index])
        reduced_ais = AIS.reduce_ais_to_csv(ais_rows)

        # closest_ship, spiller_is_hidden, oil_pixels = get_closest_ship(f"grayscale_frames/frame_0{img_count}.png", ships_near_oil)
        closest_ship, spiller_is_hidden, oil_pixels = get_closest_ship(image, "input.csv",ships_near_oil)
        # img_count += 1
        output = ""

        if closest_ship is not None:
            # print(len(closest_ship))
            # print(f"Oil pixels: {oil_pixels}")

            existing_ship = next((ship for ship in ships_near_oil if ship["mmsi"] == closest_ship["mmsi"]), None)
            if existing_ship is None:
                closest_ship["proximity_count"] = 1
                ships_near_oil.append(closest_ship)
                print(f"New ship near oil: {closest_ship}")
            else:
                existing_ship["proximity_count"] += 1
                existing_ship["lat_px"] = closest_ship["lat_px"]
                existing_ship["lon_px"] = closest_ship["lon_px"]
                existing_ship["lon"] = closest_ship["lon"]
                existing_ship["lat"] = closest_ship["lat"]
                # print("The spilling ship is hiding it's AIS signal\n")

            # img = cv2.imread(image)
            # lat_px = int(closest_ship["lat_px"])
            # lon_px = int(closest_ship["lon_px"])
            # cv2.circle(img, (lon_px, lat_px), 8, (0, 0, 255), -1)
            # cv2.imshow("Live Feed", img)

            top_ship = max(ships_near_oil, key=lambda s: s["proximity_count"])
            output = f"A{int(oil_pixels['estimated_area_km2'])},mmsi{top_ship['mmsi']}"
            with open("output.txt", "w") as f:
                    f.write(output)

            
        elif not oil_present:
            output = "A0,mmsi0"
            with open("output.txt", "w") as f:
                    f.write(output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if ships_near_oil:
    top_ship = max(ships_near_oil, key=lambda s: s["proximity_count"])
    print(f"Ship with highest proximity count is mmsi: {top_ship['mmsi']} with proximity_count: {top_ship['proximity_count']}, at lon: {top_ship['lon']} lat: {top_ship['lat']}")

print(ships_near_oil)

# cv2.destroyAllWindows()