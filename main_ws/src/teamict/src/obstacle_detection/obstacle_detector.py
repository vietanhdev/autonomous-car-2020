#!/usr/bin/env python
import cv2
import numpy as np
import rospy


class ObstacleDetector:

    def __init__(self, debug_stream=None):
        self.MAX_UINT8 = 255
        self.counter = 0
        self.area_130 = 130
        self.area_100 = 110
        self.debug_stream = debug_stream

        # Initialize debug stream
        if self.debug_stream:
            self.debug_stream.create_stream('obstacle_detector', 'debug/obstacle_detector')

    def ground(self, gray_img, x, y, n, T1, T2):
        if gray_img[y][x] > T1 or gray_img[y][x] < T2:
            return 0
        if int(gray_img[y][x]) - int(gray_img[y + n][x]) >= 1:
            for i in range(n):
                if int(gray_img[y + i][x]) - int(gray_img[y + i + 1][x]) < 0:
                    return gray_img[y][x]
            return 0
        else:
            return gray_img[y][x]

    def resize_np(self, img_np, percent):
        h, w = img_np.shape
        w = int(w * percent)
        h = int(h * percent)
        resized_img = cv2.resize(img_np, (w, h))
        return resized_img

    def find_nearest_object(self, bbox):
        b_new = list(bbox[i][1] + bbox[i][3] for i in range(len(bbox)))
        index = np.argmax(np.array(b_new))
        nearest_obstacle = bbox[index]
        return nearest_obstacle

    def regress_danger_zone(self, obstacle_left, obstacle_right):
        # danger zone
        danger_zone = (0, 0)  # init
        danger_zone_y = 0
        # 2 objects
        if obstacle_left != 0 and obstacle_right != 0:
            (x_left, y_left, w_left, h_left) = obstacle_left
            (x_right, y_right, w_right, h_right) = obstacle_right

            # remove the further one if not parallel
            if (y_left + h_left) - (y_right + h_right) > 50:
                obstacle_right = 0
                y = y_left
            elif (y_right + h_right) - (y_left + h_left) > 50:
                obstacle_left = 0
                y = y_right

            # go through between them
            else:
                right_edge_object_left = x_left + w_left
                left_edge_object_right = x_right

                center_zone = int((right_edge_object_left + left_edge_object_right) / 2)
                danger_zone = (-1, center_zone)

        # single object
        if obstacle_left != 0 and obstacle_right == 0:
            (x, y, w, h) = obstacle_left
            center_object = int((x + x + w) / 2)
            if 100 < center_object < 220:
                danger_zone = (x - self.area_100, x + w + self.area_100)
            else:
                danger_zone = (x - self.area_130, x + w + self.area_130)
            danger_zone_y = y + h - 1
        if obstacle_right != 0 and obstacle_left == 0:
            (x, y, w, h) = obstacle_right
            center_object = int((x + x + w) / 2)
            if 100 < center_object < 220:
                danger_zone = (x - self.area_100, x + w + self.area_100)
            else:
                danger_zone = (x - self.area_130, x + w + self.area_130)
            danger_zone_y = y + h - 1
        return danger_zone, danger_zone_y

    def find_danger_zone(self, car_mask, perdestrian_mask, min_width=20, min_height=20):

        img_np = car_mask | perdestrian_mask
        gray_uint8 = img_np
        if cv2.getVersionMajor() in [2, 4]:
            # OpenCV 2, OpenCV 4 case
            contours, hierarchy = cv2.findContours(gray_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # OpenCV 3 case
            _, contours, hierarchy = cv2.findContours(gray_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_RGB_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        # img_RGB_np = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        # cv2.drawContours(img_RGB_np, contours, -1, (MAX_UINT16, 0, 0), 2)
        # cv2.imshow('contours', img_RGB_np)

        bbox_left = []
        bbox_right = []
        # print('number of contours', len(contours))
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            # print(w,h)
            if h > min_height:
                cv2.rectangle(img_RGB_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # draw danger zone
                cv2.rectangle(img_RGB_np, (x - self.area_100, y), (x + w + self.area_100, y + h),
                              (self.MAX_UINT8, 0, 0), 2)

                center_x = x + int(w / 2)

                if center_x < 160:
                    bbox_left.append((x, y, w, h))
                else:
                    bbox_right.append((x, y, w, h))

        # left - right
        obstacle_left = obstacle_right = 0
        if len(bbox_left) > 0:
            obstacle_left = self.find_nearest_object(bbox_left)
        if len(bbox_right) > 0:
            obstacle_right = self.find_nearest_object(bbox_right)

        danger_zone = self.regress_danger_zone(obstacle_left, obstacle_right)

        if self.debug_stream:
            self.debug_stream.update_image('obstacle_detector', img_RGB_np)

        return danger_zone