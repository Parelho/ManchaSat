import cv2
import numpy as np

class CalcSpillArea:
    def __init__(self, crop_offsets=(75, 45, 20, 100)):
        """
        crop_offsets: (left, right, top, bottom)
        """
        self.crop_offsets = crop_offsets

    def crop_center(self, frame):
        """Crop the image based on offsets (left, right, top, bottom)."""
        h, w, _ = frame.shape
        left, right, top, bottom = self.crop_offsets
        return frame[top:h - bottom, left:w - right]

    def analyze(self, frame, oil_pixels, save_resized_path="resized_input.jpg"):
        """
        Takes the original frame and total oil_pixels (from detection),
        crops and resizes the image, and returns area estimations.

        Returns a dict with:
        - oil_pixels
        - total_pixels
        - oil_percentage
        - estimated_area_km2
        """
        # Crop and resize
        cropped = self.crop_center(frame)
        resized_cropped = cv2.resize(cropped, (1920, 1080))
        # cv2.imwrite(save_resized_path, resized_cropped)

        total_pixels = cropped.shape[0] * cropped.shape[1]
        percentage = oil_pixels / total_pixels

        # --- Area estimation ---
        lat_pixels_per_degree = 1080 / 180
        lon_pixels_per_degree = 1920 / 360
        km_per_degree_lat = 111  # average ~111 km per degree latitude
        km_per_degree_lon = 111

        km_per_pixel_lat = km_per_degree_lat / lat_pixels_per_degree
        km_per_pixel_lon = km_per_degree_lon / lon_pixels_per_degree
        km_per_pixel = (km_per_pixel_lat + km_per_pixel_lon) / 2

        area_km2 = oil_pixels * (km_per_pixel ** 2)

        return {
            "oil_pixels": int(oil_pixels),
            "total_pixels": int(total_pixels),
            "oil_percentage": float(percentage),
            "estimated_area_km2": float(area_km2)
        }
