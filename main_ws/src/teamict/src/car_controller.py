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
from simple_pid import PID

class CarController:

    def __init__(self, segmentation, debug_stream=None):

        rospy.init_node(config.TEAM_NAME, anonymous=True)
        self.speed_pub = rospy.Publisher(config.TOPIC_SET_SPEED, Float32, queue_size=1)
        self.steer_angle_pub = rospy.Publisher(config.TOPIC_SET_ANGLE, Float32, queue_size=1)

        self.segmentation = segmentation
        self.obstacle_detector = ObstacleDetector(debug_stream=debug_stream)

        self.h, self.w = 240, 320
        self.debug_stream = debug_stream

        self.current_traffic_sign = 0

        # Turning
        self.is_turning = False
        self.turning_time_begin = -1
        self.current_turning_direction = 0
        self.sign_count = 0

        # Object avoidance
        self.object_avoidance_direction = 0
        self.last_object_time = 0

        # Initialize debug stream
        if self.debug_stream:
            self.debug_stream.create_stream('car_controlling', 'debug/car_controlling')

        # Subcribe to traffic sign topic
        rospy.Subscriber("/teamict/trafficsign", Int32, self.new_traffic_sign_callback)

        # PID controller
        self.steer_pid = PID(1, 0.2, 0.01, setpoint=0)


    def new_traffic_sign_callback(self, data):
        self.current_traffic_sign = int(data.data)

        if self.current_traffic_sign == config.SIGN_NO_SIGN:
            return
        
        if self.current_traffic_sign == config.SIGN_LEFT:
            sign = "LEFT"
        else:
            sign = "RIGHT"

        print("Traffic sign detected: {}".format(sign))

    def control(self, img):
        """
        Calculate steering angle
        :param img: bgr image to get road mask
        :return: np.float32 binary image (1-road, 0-other)
        """

        vis_img = None

        # Find steer angle
        steer_angle, vis_img = self.cal_steer_angle(img, vis_img=img)

        steer_angle *= 0.6 # Reduce steering angle to get smooth turning

        steer_angle = self.steer_pid(-steer_angle)

        if abs(steer_angle) > config.MAX_STEER_ANGLE:
            steer_angle = config.MAX_STEER_ANGLE * abs(steer_angle) / steer_angle

        if not rospy.is_shutdown():
            self.current_speed = config.MAX_SPEED - 2.5 * abs(steer_angle) - abs(self.object_avoidance_direction) * config.OBSTACLE_SPEED_DECAY

            if self.current_speed > config.MAX_SPEED:
                self.current_speed = config.MAX_SPEED

            if self.current_speed < config.MIN_SPEED:
                self.current_speed = config.MIN_SPEED

            self.speed_pub.publish(self.current_speed)
            self.steer_angle_pub.publish(steer_angle)

        if config.SHOW_VIS_DEBUG_IMAGE:

            # Draw angle
            half_car_width = config.CAR_WIDTH // 2 
            cv2.arrowedLine(vis_img, (self.w // 2, self.h - 30), (self.w // 2 + int(steer_angle * 4), self.h * 2 // 3),  (0, 255, 0), 3)

            # Draw speed
            vis_img = cv2.putText(vis_img, str(int(self.current_speed)) + " km/h", (self.w // 2 - 40, self.h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw traffic sign
            if self.current_traffic_sign != config.SIGN_NO_SIGN:
                if self.current_traffic_sign == config.SIGN_LEFT:
                    sign = "<="
                else:
                    sign = "=>"
                vis_img = cv2.circle(vis_img, (self.w - 30, self.h - 30), 22, (255,255,255), -1)
                vis_img = cv2.circle(vis_img, (self.w - 30, self.h - 30), 20, (255,125,0), -1)
                vis_img = cv2.putText(vis_img, sign, (self.w - 45, self.h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw danger zone
            if self.object_avoidance_direction == 1:
                vis_img = cv2.circle(vis_img, (self.w // 2 - 60, self.h - 30), 12, (255,255,255), -1)
                vis_img = cv2.circle(vis_img, (self.w // 2 - 60, self.h - 30), 10, (0,255,255), -1)
                vis_img = cv2.circle(vis_img, (self.w // 2 - 60, self.h - 30), 4, (0,0,255), -1)

            if self.object_avoidance_direction == -1:
                vis_img = cv2.circle(vis_img, (self.w // 2 + 60, self.h - 30), 12, (255,255,255), -1)
                vis_img = cv2.circle(vis_img, (self.w // 2 + 60, self.h - 30), 10, (0,255,255), -1)
                vis_img = cv2.circle(vis_img, (self.w // 2 + 60, self.h - 30), 4, (0,0,255), -1)

            # Draw speed
            # cv2.imshow("vis_img - Team ICT", vis_img)
            # cv2.waitKey(1)


    def cal_steer_angle(self, img, vis_img=None):
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

        if vis_img is not None:
            vis_img = self.segmentation.get_visualization_img(vis_img, seg_masks)

        # Get road mask
        road_mask = seg_masks[TrafficObject.ROAD.name]

        # Filter to get only largest white area in road mask
        if cv2.getVersionMajor() in [2, 4]:
            # OpenCV 2, OpenCV 4 case
            contours, hierarchy = cv2.findContours(road_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # OpenCV 3 case
            _, contours, hierarchy = cv2.findContours(road_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Choose largest contour
        best = -1
        maxsize = -1
        count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > maxsize :
                maxsize = cv2.contourArea(cnt)
                best = count
            count = count + 1

        road_mask[:, :] = 0
        if best != -1:
            cv2.drawContours(road_mask,[contours[best]], 0, 255, -1)

        # cv2.imshow("Debug", road_mask)
        # cv2.waitKey(1)

        car_mask = seg_masks[TrafficObject.CAR.name]
        perdestrian_mask = seg_masks[TrafficObject.PERDESTRIAN.name]
        road_mask = road_mask & cv2.bitwise_not(car_mask)

        # Convert to bird view
        road_mask_bv = self.bird_view(road_mask)

        # ====== Turning =======

        if self.is_turning:
            if self.turning_time_begin + config.TURNING_TIME < time.time():
                self.is_turning = False
            else:
                return self.current_turning_direction * config.TURNING_ANGLE, vis_img
        else:
            interested_area = road_mask_bv[80:180, :]
            lane_area = np.count_nonzero(interested_area)

            if lane_area > config.ROAD_AREA_TO_TURN:
                print("Turning")
                self.is_turning = True
                self.turning_time_begin = time.time()

                self.current_turning_direction = self.current_traffic_sign
                print(self.current_turning_direction)

                # Reset traffic sign
                self.current_traffic_sign = config.SIGN_NO_SIGN

                return self.current_turning_direction * config.TURNING_ANGLE, vis_img
                
        # ====== If not turning, calculate steering angle using middle point =======

        # TODO: The method to calculate the middle point and angle now is so simple.
        # Research for others in the future
        interested_row = road_mask_bv[int(road_mask_bv.shape[0] / 3 * 2), :].reshape((-1,))
        white_pixels = np.argwhere(interested_row > 0)

        if white_pixels.size != 0:
            middle_pos = np.mean(white_pixels)
        else:
            middle_pos = 160

        if middle_pos != middle_pos: # is NaN
            middle_pos = 0



        # ====== Obstacle avoidance =======
        danger_zone, danger_zone_y = self.obstacle_detector.find_danger_zone(car_mask, perdestrian_mask)

        # print(danger_zone, danger_zone_y)

        # Avoid obstacles
        if danger_zone != (0, 0):

            # 2 objects
            if danger_zone[0] == -1:
                self.object_avoidance_direction = 0
                # middle_pos = danger_zone[1]

            # single object
            else:
                center_danger_zone = int((danger_zone[0] + danger_zone[1]) / 2)
                
                count_road_pixels_left = np.count_nonzero(road_mask[danger_zone_y, :center_danger_zone])
                count_road_pixels_right = np.count_nonzero(road_mask[danger_zone_y, center_danger_zone:])

                # obstacle is on the right
                if count_road_pixels_left > count_road_pixels_right:
                    self.object_avoidance_direction = -1
                    self.last_object_time = time.time()
                    # middle_pos = danger_zone[0]
                    print("OBSTACLE: RIGHT")
                # left
                elif count_road_pixels_left < count_road_pixels_right:
                    self.object_avoidance_direction = 1
                    self.last_object_time = time.time()
                    # middle_pos = danger_zone[1]
                    print("OBSTACLE: LEFT")


        # Object avoidance
        if self.last_object_time > time.time() - config.OBSTACLE_AVOIDANCE_TIME:
            middle_pos += config.OBSTACLE_AVOIDANCE_OFFSET * self.object_avoidance_direction
            print("Obstacle avoidance direction: " + str(self.object_avoidance_direction))
        elif self.last_object_time < time.time() - config.OBSTACLE_AVOIDANCE_TIME - 1:
            self.object_avoidance_direction = 0
            # print("Obstacle was over")


        if self.debug_stream:
            half_car_width = config.CAR_WIDTH // 2 
            cv2.line(img_bv, (int(middle_pos), self.h // 2), (self.w // 2, self.h), (255, 0, 0), 2)
            cv2.line(img_bv, (int(middle_pos) + half_car_width, self.h // 2), (self.w // 2 + half_car_width, self.h), (255, 0, 255), 3)
            cv2.line(img_bv, (int(middle_pos) - half_car_width, self.h // 2), (self.w // 2 - half_car_width, self.h), (255, 0, 255), 3)
            self.debug_stream.update_image('car_controlling', img_bv)  


        # Add offset to middle pos
        middle_pos += config.MIDDLE_POS_OFFSET

        # Distance between MiddlePos and CarPos
        distance_x = middle_pos - self.w / 2
        distance_y = self.h - self.h / 3 * 2

        # Angle to middle position
        steer_angle = math.atan(float(distance_x) / distance_y) * 180 / math.pi * 1.2

        # QIK MATH
        # steer_angle = ((middle_pos - 160) / 160) * 60

        return steer_angle, vis_img

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
