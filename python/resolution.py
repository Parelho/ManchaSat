import math

class Resolution:
    def __init__(self) -> None:
        pass

    def get_resolution(distance_to_earth, horizontal_fov, vertical_fov):
        horizontal_resolution = ((distance_to_earth * math.sin(math.radians(horizontal_fov/2))) / math.sin(math.radians(180-(90+horizontal_fov/2)))) * 2
        vertical_resolution = ((distance_to_earth * math.sin(math.radians(vertical_fov/2))) / math.sin(math.radians(180-(90+vertical_fov/2)))) * 2
        
        return horizontal_resolution, vertical_resolution