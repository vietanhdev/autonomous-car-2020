#!/usr/bin/env python

import cv2
import math
import numpy as np
import rospkg
import rospy
from std_msgs.msg import Float32, String, Bool, Int32
import config
import time
from semantic_segmentation.traffic_objects import TrafficObject, OBJECT_COLORS
from obstacle_detection.obstacle_detector import ObstacleDetector

class CarController:

    def __init__(self, segmentation, debug_stream=None):

        rospy.init_node(config.TEAM_NAME, anonymous=True)
        self.speed_pub = rospy.Publisher(config.TOPIC_SET_SPEED, Float32, queue_size=1)
        self.steer_angle_pub = rospy.Publisher(config.TOPIC_SET_ANGLE, Float32, queue_size=1)

        self.segmentation = segmentation
        self.obstacle_detector = ObstacleDetector(debug_stream=debug_stream)

        self.current_speed = 40
        self.h, self.w = 240, 320
        self.debug_stream = debug_stream

        self.current_traffic_sign = 0

        # Turning
        self.is_turning = False
        self.turning_time_begin = -1
        self.current_turning_direction = 1
        self.sign_count = 0

        # Initialize debug stream
        if self.debug_stream:
            self.debug_stream.create_stream('car_controlling', 'debug/car_controlling')

        # Subcribe to traffic sign topic
        rospy.Subscriber("/teamict/trafficsign", Int32, self.new_traffic_sign_callback)


    def new_traffic_sign_callback(self, data):
        traffic_sign = int(data.data)
        if traffic_sign == 0:
            self.current_traffic_sign = -1
        else:
            self.current_traffic_sign = 1

    def get_next_direction(self):
        return self.current_traffic_sign

    def control(self, img):
        """
        Calculate steering angle
        :param img: bgr image to get road mask
        :return: np.float32 binary image (1-road, 0-other)
        """

        # Find steer angle
        steer_angle = self.cal_steer_angle(img)
        speed = 0

        if not rospy.is_shutdown():

            if self.current_speed >= config.MIN_SPEED:
                speed = max(config.MIN_SPEED,
                            self.current_speed - config.SPEED_DECAY * (config.BASE_SPEED - config.MIN_SPEED) * abs(steer_angle ** 2) / (config.MAX_STEER_ANGLE ** 2))

                if self.current_speed < config.BASE_SPEED and slow_down == 0:
                    self.current_speed += 0.4

            self.speed_pub.publish(speed)
            self.steer_angle_pub.publish(steer_angle * 0.6)


        return self.is_turning, steer_angle, speed

    def cal_steer_angle(self, img):
        """
        Calculate steering angle for car
        :param img: bgr image
        :return: steering angle (-60 to 60)
        """
        # Init steering angle to 0
        steer_angle = 0

        # Get birdview image
        img_bv = self.bird_view(img)

        # Run semantic segmentation on RGB image
        seg_masks = self.segmentation.get_masks(img)

        # Get road mask
        road_mask = seg_masks[TrafficObject.ROAD.name]

        # Convert to bird view
        road_mask_bv = self.bird_view(road_mask)

        # ====== Turning =======

        if self.is_turning:
            if self.turning_time_begin + config.TURNING_TIME < time.time():
                self.is_turning = False
            else:
                return self.current_turning_direction * config.TURNING_ANGLE
        else:
            interested_area = road_mask_bv[80:180, :]
            lane_area = np.count_nonzero(interested_area)

            if lane_area > 12000:
                print("Turning")
                self.is_turning = True
                self.current_turning_direction = self.get_next_direction()
                self.turning_time_begin = time.time()

                # Reset traffic sign
                self.current_traffic_sign = 0

                return self.current_turning_direction * config.TURNING_ANGLE
                
        # ====== If not turning, calculate steering angle using middle point =======


        # TODO: The method to calculate the middle point and angle now is so simple.
        # Research for others in the future
        interested_row = road_mask_bv[road_mask_bv.shape[0] / 3 * 2, :].reshape((-1,))
        white_pixels = np.argwhere(interested_row > 0)

        if white_pixels.size != 0:
            middle_pos = np.mean(white_pixels)
        else:
            middle_pos = 160

        if middle_pos != middle_pos: # is NaN
            middle_pos = 0



        # ====== Obstacle avoidance =======
        # Get masks
        car_mask = seg_masks[TrafficObject.CAR.name]
        perdestrian_mask = seg_masks[TrafficObject.PERDESTRIAN.name]
        danger_zone = self.obstacle_detector.find_danger_zone(car_mask, perdestrian_mask)

        # Avoid obstacles
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

        # if middle_pos > 640:
        #     middle_pos = 640
        # if middle_pos < -320:
        #     middle_pos = -320


        if self.debug_stream:
            half_car_width = config.CAR_WIDTH // 2 
            cv2.line(img_bv, (int(middle_pos), self.h / 2), (self.w / 2, self.h), (255, 0, 0), 2)
            cv2.line(img_bv, (int(middle_pos) + half_car_width, self.h / 2), (self.w / 2 + half_car_width, self.h), (255, 0, 255), 3)
            cv2.line(img_bv, (int(middle_pos) - half_car_width, self.h / 2), (self.w / 2 - half_car_width, self.h), (255, 0, 255), 3)
            self.debug_stream.update_image('car_controlling', img_bv)
            

        # Distance between MiddlePos and CarPos
        distance_x = middle_pos - self.w / 2
        distance_y = self.h - self.h / 3 * 2

        # Angle to middle position
        steer_angle = math.atan(float(distance_x) / distance_y) * 180 / math.pi

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
