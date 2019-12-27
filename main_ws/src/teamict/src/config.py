# -*- coding: utf-8 -*- 

import rospkg
from os import path
import json

TEAM_NAME = 'teamict'
PACKAGE_NAME = 'teamict'

# ===== Development mode of the system. Set it to True for debug streams =====
DEVELOPMENT = True
SHOW_VIS_DEBUG_IMAGE = True # Show debug image for demonstration

# ===== Package & Data folder (for JSON config. and model files) =====
PACKAGE_PATH = rospkg.RosPack().get_path(PACKAGE_NAME)
DATA_FOLDER = path.join(PACKAGE_PATH, 'data/')

# ===== Read JSON configuration files =====
SEMANTIC_SEGMENTATION_CONFIG_PATH = path.join(DATA_FOLDER, 'semantic_seg_ENet.conf.json')
SIGN_DETECTION_CONFIG_PATH =  path.join(DATA_FOLDER, 'sign_detector.conf.json')

with open(SEMANTIC_SEGMENTATION_CONFIG_PATH) as config_buffer:
    SEMANTIC_SEGMENTATION_CONFIG = json.loads(config_buffer.read())
with open(SIGN_DETECTION_CONFIG_PATH) as config_buffer:
    SIGN_DETECTION_CONFIG = json.loads(config_buffer.read())


# ===== ROS Topic names =====

USE_OLD_SIMULATOR = False

# /tênđội/set_speed: Topic được publish từ ROS_node được định nghĩa dưới dạng số thực (Float32). Là tốc độ xe cần đạt. ( Mặc định đang để là /team1/set_speed, nếu nhập tên đội khác, cần sửa lại topic trong code /lane_detect/src/main.cpp theo tên đội đã nhập ở app)
if USE_OLD_SIMULATOR:
    TOPIC_SET_SPEED = '/{}_speed'.format(TEAM_NAME)
else:
    TOPIC_SET_SPEED = '/{}/set_speed'.format(TEAM_NAME)

# /tênđội/set_angle: Topic được publish từ ROS_node định nghĩa dưới dạng số thực (Float32). Truyền góc lái của xe. ( Mặc định đang để là /team1/set_angle, nếu nhập tên đội khác, cần sửa lại topic trong code /lane_detect/src/carcontrol.cpp theo tên đội đã nhập ở app)
if USE_OLD_SIMULATOR:
    TOPIC_SET_ANGLE = '/{}_steerAngle'.format(TEAM_NAME)
else:
    TOPIC_SET_ANGLE = '/{}/set_angle'.format(TEAM_NAME)

# /tênđội/set_camera_angle: Topic được publish từ ROS_node định nghĩa dưới dạng số thực (Float32). Truyền quay của camera.
TOPIC_SET_CAMERA_ANGLE = '/{}/set_camera_angle'.format(TEAM_NAME)

# /tênđội/camera/rgb/compressed: Topic dùng để subcribe ảnh rgb thu được trên xe. Ảnh thu được là ảnh nén theo chuẩn “img”.( Mặc định đang để là /team1/camera/rgb/compressed, nếu nhập tên đội khác, cần sửa lại topic trong code /lane_detect/src/carcontrol.cpp theo tên đội đã nhập ở app)
if USE_OLD_SIMULATOR:
    TOPIC_GET_IMAGE = '/{}_image/compressed'.format(TEAM_NAME)
else:
    TOPIC_GET_IMAGE = '/{}/camera/rgb/compressed'.format(TEAM_NAME)

# /tênđội/camera/depth/compressed: Topic dùng để subcribe ảnh depth thu được trên xe. Ảnh thu được là ảnh nén theo chuẩn “img”.( Mặc định đang để là /team1/camera/depth/compressed, nếu nhập tên đội khác, cần sửa lại topic trong code /lane_detect/src/carcontrol.cpp theo tên đội đã nhập ở app)
TOPIC_GET_DEPTH_IMAGE = '/{}/camera/depth/compressed'.format(TEAM_NAME)

IMAGE_QUEUE_SIZE = 3

# ===== Car properties =====
CAR_WIDTH = 30 # In pixels
MIN_SPEED = 20
MAX_SPEED = 50
SPEED_DECAY = 2
MAX_STEER_ANGLE = 50.0

STEER_ANGLE_SCALE = 1

MIDDLE_POS_OFFSET = 0
MIDDLE_POS_SCALE = 1

SPEED_SLOW_DOWN = 1


# ===== Traffic sign =====
# Turning time
ROAD_AREA_TO_TURN = 12000
TURNING_TIME = 1
TURNING_ANGLE = 40

# Traffic sign detection
TRAFFIC_SIGN_DETECTION_THRESHOLD = 0.6
SIGN_NO_SIGN = 0
SIGN_LEFT = -1
SIGN_RIGHT = 1

SIGN_HISTORY_LENGTH = 10
SIGN_EXP_TIME = 5
SIGN_DETECTION_THRESHOLD = 3 # Number of signs for a conclusion


# ===== Obstacle avoidance =====
OBSTACLE_AVOIDANCE_TIME = 3
OBSTACLE_AVOIDANCE_OFFSET = 10 # Center point offset to avoid obstacle
OBSTACLE_SPEED_DECAY = 10