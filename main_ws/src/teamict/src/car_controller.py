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
from param import Param

class CarController:

    def __init__(self, lane_detector):
        rospy.init_node(config.TEAM_NAME, anonymous=True)
        self.speed_pub = rospy.Publisher(config.TOPIC_SET_SPEED, Float32, queue_size=1)
        self.steer_angle_pub = rospy.Publisher(config.TOPIC_SET_ANGLE, Float32, queue_size=1)
        self.param = Param()
        self.lane_detector = lane_detector
        self.current_speed = 60
        self.is_turning = 0
        self.h, self.w = 240, 320
        rospy.Rate(10)

    def control(self, img, sign, is_go, danger_zone, slow_down):

        # Find steer angle
        steer_angle = self.cal_steer_angle(img, sign, danger_zone)
        speed = 0

        if not rospy.is_shutdown():

            if self.current_speed >= self.param.min_speed:
                speed = max(self.param.min_speed,
                            self.current_speed - self.param.speed_decay * (self.param.base_speed - self.param.min_speed) * abs(steer_angle ** 2) / (self.param.max_steer_angle ** 2))

                if self.current_speed < self.param.base_speed and slow_down == 0:
                    self.current_speed += 0.4

            self.speed_pub.publish(speed)
            if is_go:
                self.steer_angle_pub.publish(steer_angle * 0.6)


        return self.is_turning, steer_angle, speed

    def cal_steer_angle(self, img, sign, danger_zone):

        steer_angle = 0

        img_bv = self.bird_view(img)

        road_mask = self.lane_detector.get_road_mask(img)

        # cv2.imshow("road_mask", road_mask)

        interested_row = road_mask[120, :].reshape((-1,))
        middle_pos = np.mean(np.argwhere(interested_row > 0))

        if middle_pos != middle_pos: # is NaN
            middle_pos = 0


        # avoid obstacles
        # if danger_zone != (0, 0):
        #     # 2 objects
        #     if danger_zone[0] == -1:
        #         middle_pos = danger_zone[1]
        #     # single object
        #     else:
        #         center_danger_zone = int((danger_zone[0] + danger_zone[1]) / 2)
        #         # print(danger_zone, center_danger_zone)
        #         if danger_zone[0] + 20 < middle_pos < danger_zone[1] - 20:
        #             # obstacle's on the right
        #             if (middle_pos - 160) * 1 + 160 < center_danger_zone:
        #                 print("on the right")
        #                 middle_pos = danger_zone[0]
        #             # left
        #             else:
        #                 print("on the left")
        #                 middle_pos = danger_zone[1]

        if middle_pos > 640:
            middle_pos = 640
        if middle_pos < -320:
            middle_pos = -320

        cv2.line(img_bv, (int(middle_pos), self.h / 2), (self.w / 2, self.h), (255, 0, 0), 2)
        # cv2.imshow("Bird view", img_bv[:, :, :])

        # Distance between MiddlePos and CarPos
        distance_x = middle_pos - self.w / 2
        distance_y = self.h - self.h / 3 * 2

        # print(middle_pos)

        # Angle to middle position
        steer_angle = math.atan(float(distance_x) / distance_y) * 180 / math.pi
        cv2.waitKey(1)

        # QIK MATH
        # steer_angle = ((middle_pos - 160) / 160) * 60

        return steer_angle

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
