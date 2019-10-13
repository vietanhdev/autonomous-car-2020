#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import time
import config
import roslib
import rospy
import cv2
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
from param import Param
import geometry_msgs.msg
import threading
from debug_stream import DebugStream 


class ImageProcessor:

    def __init__(self, cv_bridge, debug_stream):
        '''
        Init ros node to receive images
        '''
        # ROS message exchange rate
        self.rate = rospy.Rate(10)
        self.cv_bridge = cv_bridge
        self.debug_stream = debug_stream

        # Setup pub/sub
        self.rgb_camera_sub = rospy.Subscriber(config.TOPIC_GET_IMAGE, CompressedImage, callback=self.callback_rgb_image, queue_size=1)
        self.depth_camera_sub = rospy.Subscriber(config.TOPIC_GET_DEPTH_IMAGE, CompressedImage, callback=self.callback_depth_image, queue_size=1)

        self.debug_stream.create_stream('rgb', 'debug/rgb')
        self.debug_stream.create_stream('depth', 'debug/depth')


    def callback_rgb_image(self, data):
        '''
        Function to process rgb images
        '''
        global rgb_image, rgb_image_mutex
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # image_np = self.bridge.imgmsg_to_cv2(data)

            # NOTE: image_np.shape = (240, 320, 3)
            image_np = cv2.resize(image_np, (320, 240))

            self.debug_stream.update_image('rgb', image_np)

        except CvBridgeError as e:
            print(e)


    def callback_depth_image(self, data):
        '''
        Function to process depth images
        '''
        global depth_image, depth_image_mutex
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # image_np = self.bridge.imgmsg_to_cv2(data)

            # NOTE: image_np.shape = (240, 320, 3)
            image_np = cv2.resize(image_np, (320, 240))

            self.debug_stream.update_image('depth', image_np)

            # print(self.danger_zone)
        except CvBridgeError as e:
            print(e)
