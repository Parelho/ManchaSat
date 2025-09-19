import pyais
import random

class AIS:
    def __init__(self) -> None:
        pass

    def checksum(nmea):
        data, given_checksum = nmea.strip().split('*')
        given_checksum = int(given_checksum, 16)
        calc_checksum = 0
        for c in data[1:]:
            calc_checksum ^= ord(c) # xor between given checksum and actual data

        if calc_checksum == given_checksum:
            return True
        
        return False
    
    def decode(nmea):
        return pyais.decode(nmea)
    
    def simulate_ais():
        # bits 0-5 Message ID, which can be 1,2,3
        message_id = random.randint(1,3)

        # bits 6-7 Repeat indicator
        repeat_indicator = 0

        # bits 8-37 User ID
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

        # bits 61-88 Longitude
        longitude = random.uniform(-180, 180)

        # bits 89-115 Latitude
        latitude = random.uniform(-90, 90)

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