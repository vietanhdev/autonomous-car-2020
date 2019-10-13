import config
import rospkg
path = rospkg.RosPack().get_path(config.TEAM_NAME)

from enum import Enum
class TrafficSign(Enum):
    NONE = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


class SignDetector:
    def __init__(self):
        pass
    def detect(self, img):
        pass