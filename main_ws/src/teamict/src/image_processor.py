#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import time
import config
import roslib
import rospy
import rospkg
import cv2
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
import geometry_msgs.msg
import threading
from debug_stream import DebugStream 
from os import path
from lane_detection.lane_detector import LaneDetector
from sign_detection.sign_detector import SignDetector
from depth_processor import DepthProcessor
from car_controller import CarController

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    
from Queue import Queue

PACKAGE_PATH = rospkg.RosPack().get_path(config.PACKAGE_NAME)
DATA_FOLDER = path.join(PACKAGE_PATH, 'data/')

class ImageProcessor:

    def __init__(self, cv_bridge, debug_stream=None):
        '''
        Init ros node to receive images
        '''
        
        self.cv_bridge = cv_bridge
        self.debug_stream = debug_stream

        self.sd_counter = 0
        self.cc_counter = 0
        self.dc_counter = 0
        self.bt_counter = 0
        self.st_counter = 0

        self.is_turning = False

        self.is_go = True
        self.sign = 0
        self.bbox_obstacles = []
        self.danger_zone = (0, 0)

        if self.debug_stream:
            self.debug_stream.create_stream('rgb', 'debug/rgb')
            self.debug_stream.create_stream('depth', 'debug/depth')

        # ================ Initialize controlling models ================ 

        # Lane detection
        lane_config = path.join(DATA_FOLDER, 'lane_detector.conf.json')
        lane_model = path.join(DATA_FOLDER, 'lane_detect_UNet.h5')
        self.lane_detector = LaneDetector(lane_config, lane_model, debug_stream=debug_stream)

        # Depth processor
        self.depth_processor = DepthProcessor()

        # Car controlling
        self.car_controller = CarController(self.lane_detector)

        # Setup pub/sub
        # WTF BUG!!! https://answers.ros.org/question/220502/image-subscriber-lag-despite-queue-1/
        self.rgb_camera_sub = rospy.Subscriber(config.TOPIC_GET_IMAGE, CompressedImage, callback=self.callback_rgb_image, queue_size=1, buff_size=2**24)
        self.rgb_camera_sub_2 = rospy.Subscriber(config.TOPIC_GET_IMAGE, CompressedImage, callback=self.callback_detect_sign, queue_size=1, buff_size=2**24)
        self.depth_camera_sub = rospy.Subscriber(config.TOPIC_GET_DEPTH_IMAGE, CompressedImage, callback=self.callback_detect_obstacle, queue_size=1, buff_size=2**24)
      

    def callback_detect_sign(self, data):
        pass


    def callback_detect_obstacle(self, data):

        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            image_np = cv2.resize(image_np, (320, 240))
            image_np = image_np[100:, :]
    
            self.danger_zone = self.depth_processor.pre_processing_depth_img(image_np)

            # print(self.danger_zone)
        except CvBridgeError as e:
            print(e)


    def callback_rgb_image(self, data):
        '''
        Function to process rgb images
        '''
        global rgb_image, rgb_image_mutex
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # NOTE: image_np.shape = (240, 320, 3)
            image_np = cv2.resize(image_np, (320, 240))

            # Put frame into rgb queue
            # self.rgb_frames.put(image_np)

            self.slow_down = True
            self.is_turning, steer_angle, speed = self.car_controller.control(image_np, self.sign, self.is_go, self.danger_zone, self.slow_down)

            if self.debug_stream:
                self.debug_stream.update_image('rgb', image_np)

        except CvBridgeError as e:
            print(e)


    # def callback_depth_image(self, data):
    #     '''
    #     Function to process depth images
    #     '''
    #     global depth_image, depth_image_mutex
    #     try:
    #         np_arr = np.fromstring(data.data, np.uint8)
    #         image_np = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    #         # NOTE: image_np.shape = (240, 320, 3)
    #         image_np = cv2.resize(image_np, (320, 240))

    #         # Put frame into depth queue
    #         # self.depth_frames.put(image_np)
    #         self.depth_processor.pre_processing_depth_img(image_np)

    #         if self.debug_stream:
    #             self.debug_stream.update_image('depth', image_np)

    #     except CvBridgeError as e:
    #         print(e)
