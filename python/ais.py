import pyais
import random
import math
from resolution import Resolution as rs

class AIS:
    tracked_ships = {}  # MMSI -> last known pixel coordinates
    distance_threshold = 20  # pixels

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def get_mmsi_for_ship(px, py):
        for mmsi, (last_x, last_y) in AIS.tracked_ships.items():
            distance = ((px - last_x)**2 + (py - last_y)**2)**0.5
            if distance < AIS.distance_threshold:
                AIS.tracked_ships[mmsi] = (px, py)
                return mmsi

        # New ship: assign a random 30-bit MMSI
        while True:
            mmsi = random.randint(0, 0x3fffffff)
            if mmsi not in AIS.tracked_ships:
                break

        AIS.tracked_ships[mmsi] = (px, py)
        return mmsi

    @staticmethod
    def checksum(nmea):
        data, given_checksum = nmea.strip().split('*')
        given_checksum = int(given_checksum, 16)
        calc_checksum = 0
        for c in data[1:]:
            calc_checksum ^= ord(c) # xor between given checksum and actual data

        if calc_checksum == given_checksum:
            return True
        
        return False
    
    @staticmethod
    def decode(nmea):
        return pyais.decode(nmea)
    
    @staticmethod
    def simulate_ais(custom_coords = False, horizontal_res = 2.14, vertical_res = 1.5, center = [0, 0], image_size = (512, 512), pixel_coord = None):
        '''
        center[longitude, latitude]
        '''
        # bits 0-5 Message ID, which can be 1,2,3
        message_id = random.randint(1,3)

        # bits 6-7 Repeat indicator
        repeat_indicator = 0

        # bits 8-37 User ID
        if pixel_coord is not None:
            px, py = pixel_coord
            mmsi = AIS.get_mmsi_for_ship(px, py)
        else:
            mmsi = random.randint(0, 0x3fffffff) # 0x3fffffff is the highest 30 bit number


        # bits 38-41 Navigational status
        # 0  under way using engine
        # 1  at anchor
        # 2  not under command
        # 3  restricted maneuverability
        # 4  constrained by her draught
        # 5  moored
        # 6  aground
        # 7  engaged in fishing
        # 8  under way sailing
        # 9  reserved for future amendment of navigational status for ships carrying DG, HS, or MP, or IMO hazard or pollutant category C, high speed craft (HSC)
        # 10 reserved for future amendment of navigational status for ships carrying dangerous goods (DG), harmful substances (HS) or marine pollutants (MP), or IMO hazard or pollutant category A, wing in ground (WIG)
        # 11 power-driven vessel towing astern
        # 12 power-driven vessel pushing ahead or towing alongside (regional use)
        # 13 RFU
        # 14 AIS-SART (active), MOB-AIS, EPIRB-AIS
        # 15 undefined = default (also used by AIS-SART, MOB-AIS and EPIRBAIS under test)
        navigational_status = random.randint(0,15)

        # bits 42-49 Rate of turn ROTAIS
        rotais = random.randint(-126, 126)

        # bits 50-59 SOG (speed over ground in 1/10 knot steps (0-102.2 knots))
        sog = random.randint(0, 1022) # 1022 = 102.2 knot or higher, 1023+ not available

        # bit 60 Position accuracy 1 <= 10m;0 > 10m
        position_accuracy = random.randint(0,1)

        # ---------- COORDS ----------
        # bits 61-88 Longitude
        longitude = 0
        # bits 89-115 Latitude
        latitude = 0
        
        lon_center, lat_center = center
        if pixel_coord is not None:
            px, py = pixel_coord

            # offset from center in pixels
            dx_px = px - image_size[0] / 2
            dy_px = py - image_size[1] / 2

            # convert pixels to km
            dx_km = dx_px * horizontal_res
            dy_km = dy_px * vertical_res

            # convert km to degrees
            km_per_deg_lat = 110.574
            km_per_deg_lon = 111.320 * math.cos(math.radians(lat_center))

            delta_lon = dx_km / km_per_deg_lon
            delta_lat = -dy_km / km_per_deg_lat  # minus because image y increases downward

            longitude = lon_center + delta_lon
            latitude = lat_center + delta_lat

        elif not custom_coords:
            longitude = random.uniform(-180, 180)

            latitude = random.uniform(-90, 90)
        else:
            total_width_km = horizontal_res * 256
            total_height_km = vertical_res * 256

            # Convert km to degrees
            km_per_deg_lat = 110.574
            km_per_deg_lon = 111.320 * math.cos(math.radians(lat_center))

            delta_lat = (total_height_km / 2) / km_per_deg_lat
            delta_lon = (total_width_km / 2) / km_per_deg_lon

            min_lat = lat_center - delta_lat
            max_lat = lat_center + delta_lat
            min_lon = lon_center - delta_lon
            max_lon = lon_center + delta_lon

            latitude = random.uniform(min_lat, max_lat)
            longitude = random.uniform(min_lon, max_lon)
        # ----------------------------------------

        # bits 116-127 COG (course over ground in 1/10)
        cog = random.randint(0, 3600)

        # bits 128-136 True heading 511 = not available
        true_heading = random.randint(0, 359)

        # bits 137-142 Time stamp, utc second when the report was generated 60 = unavaiable
        timestamp = random.randint(0, 59)

        msg = {
            "msg_type": message_id,
            "repeat_indicator": repeat_indicator,
            "mmsi": mmsi,
            "nav_status": navigational_status, 
            "rot": rotais,
            "sog": sog,
            "position_accuracy": position_accuracy,
            "lon": longitude,
            "lat": latitude,
            "cog": cog,
            "true_heading": true_heading,
            "timestamp": timestamp
        }

        return pyais.encode_dict(msg)
    
    def check_coordinates(ship, center_coords, image_size):
        distance = 430 # in km
        fov_horizontal = 65
        fov_vertical = 48

        horizontal_resolution, vertical_resolution = rs.get_resolution(distance, fov_horizontal, fov_vertical)
        horizontal_resolution = horizontal_resolution / 256 # km/px
        vertical_resolution = vertical_resolution / 256 # km/px

        lat_ais = ship["lat"]
        lon_ais = ship["lon"]
        lon_center, lat_center = center_coords
        # offset from center in pixels
        dx_px = ship["lon_px"] - image_size[0] / 2
        dy_px = ship["lat_px"] - image_size[1] / 2

        # convert pixels to km
        dx_km = dx_px * horizontal_resolution
        dy_km = dy_px * vertical_resolution
        # convert km to degrees
        km_per_deg_lat = 110.574
        km_per_deg_lon = 111.320 * math.cos(math.radians(lat_center))
        delta_lon = dx_km / km_per_deg_lon
        delta_lat = -dy_km / km_per_deg_lat  # minus because image y increases downward

        lon_camera = lon_center + delta_lon
        lat_camera = lat_center + delta_lat

        if (lon_camera * 1.1 >= lon_ais and lon_camera * 0.9 <= lon_ais) and (lat_camera * 1.1 >= lat_ais and lat_camera * 0.9 <= lat_ais):
            return [lon_camera, lat_camera]
        
        return None
    
    def lat_lon_to_px(centers, ships):
        pxs_per_lon = 5
        pxs_per_lat = 4.91
        # print(centers)

        for ship in ships:
            # if ship["mmsi"] == "990049243":
                closest_center = [-1,-1] # lon in px, lat in px
                closest_difference = 999999 # total difference from lat and lon

                # Adjustment to reconvert lon and lat to their value in pixels
                lon_adjusted = lon_adjusted = ship["lon"] * pxs_per_lon + 976.5
                lat_adjusted = 459.5 - ship["lat"] * pxs_per_lat
                # print(f"lon_adjusted: {lon_adjusted} lat_adjusted: {lat_adjusted}\n")

                for center in centers:
                    difference_lon = abs(center[0] - lon_adjusted)
                    difference_lat = abs(center[1] - lat_adjusted)
                    new_closest_difference = difference_lon + difference_lat
                    # print(f"difference_lon = {difference_lon}, difference_lat = {difference_lat}, new_closest_difference = {new_closest_difference}\n")

                    if (new_closest_difference < closest_difference):
                        closest_difference = new_closest_difference
                        closest_center = center
                
                ship["lon_px"] = closest_center[0]
                ship["lat_px"] = closest_center[1]

        return ships