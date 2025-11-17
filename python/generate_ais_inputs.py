import cv2
from tflite_runtime.interpreter import Interpreter
from image_processing import ImageProcessing as ip
import glob
import csv
import os

model_path = "model.tflite"
interpreter = Interpreter(model_path=model_path, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def get_closest_ship(image):
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
    
    return centers

# ---------- MAIN ----------
pxs_per_lon = 5
pxs_per_lat = 4.91
images = glob.glob("grayscale_frames/*.png")

for f in glob.glob("inputs/*.csv"):
    os.remove(f)

for image in images:
    centers = get_closest_ship(image)
    csv_name = image.split(".png")
    csv_name = csv_name[0].split("/")

    with open(f"inputs/{csv_name[1]}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["mmsi", "lon", "lat"])

        for i, center in enumerate(centers):
            lon_adjusted = (center[0] - 976.5) / pxs_per_lon
            lat_adjusted = (459.5 - center[1]) / pxs_per_lat
            writer.writerow([i, lon_adjusted, f"{lat_adjusted:.1f}"])