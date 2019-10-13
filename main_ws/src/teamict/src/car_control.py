#!/usr/bin/env python

import cv2
import math
import numpy as np
import rospkg
import rospy
from std_msgs.msg import Float32, String, Bool
import config
path = rospkg.RosPack().get_path(config.TEAM_NAME)
import time

class CarControl:

    def __init__(self, param):
        rospy.init_node(config.TEAM_NAME, anonymous=True)
        self.speed_pub = rospy.Publisher(config.TOPIC_SET_SPEED, Float32, queue_size=1)
        self.steer_angle_pub = rospy.Publisher(config.TOPIC_SET_ANGLE, Float32, queue_size=1)
        rospy.Rate(10)

    def unwarp(self, img, src, dst):
        h, w = img.shape[:2]
        M = cv2.getPerspectiveTransform(src, dst)

        unwarped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
        return unwarped

    def bird_view(self, source_img, isBridge=False):
        h, w = source_img.shape[:2]
        # define source and destination points for transform

        src = np.float32([(100, 120),
                          (220, 120),
                          (0, 210),
                          (320, 210)])

        dst = np.float32([(120, 0),
                          (w - 120, 0),
                          (120, h),
                          (w - 120, h)])

        src_bridge = np.float32([(50, 180),
                          (270, 180),
                          (0, 210),
                          (320, 210)])

        dst_bridge = np.float32([(80, 0),
                          (w - 80, 0),
                          (80, h),
                          (w - 80, h)])

        if isBridge == True:
            src = src_bridge
            dst = dst_bridge
        # change perspective to bird's view
        unwarped = self.unwarp(source_img, src, dst)
        return unwarped
