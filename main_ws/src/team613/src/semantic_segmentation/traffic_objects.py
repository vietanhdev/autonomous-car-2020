from enum import Enum

class TrafficObject(Enum):
    OTHER = 0
    ROAD = 1
    CAR = 3
    PERDESTRIAN = 2

OBJECT_COLORS = {
    "OTHER": (0, 0, 0),
    "ROAD": (0, 0, 255),
    "CAR": (255, 0, 0),
    "PERDESTRIAN": (0, 255, 0)
}