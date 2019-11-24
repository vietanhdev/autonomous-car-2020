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
from semantic_segmentation.semantic_segmentation import SemanticSegmentation
from obstacle_detection.depth_processor import DepthProcessor
from car_controller import CarController


PACKAGE_PATH = rospkg.RosPack().get_path(config.PACKAGE_NAME)
DATA_FOLDER = path.join(PACKAGE_PATH, 'data/')

class ImageProcessor:

    def __init__(self, cv_bridge, debug_stream=None):
        '''
        Init ros node to receive images
        '''
        
        self.cv_bridge = cv_bridge
        self.debug_stream = debug_stream

        if self.debug_stream:
            self.debug_stream.create_stream('rgb', 'debug/rgb')
            self.debug_stream.create_stream('depth', 'debug/depth')

        # ================ Initialize controlling models ================ 

        # Segmentation
        seg_config = path.join(DATA_FOLDER, 'semantic_seg_ENet.conf.json')
        self.semantic_segmentation = SemanticSegmentation(seg_config, debug_stream=debug_stream)

        # Depth processor
        self.depth_processor = DepthProcessor(debug_stream=debug_stream)

        # Car controlling
        self.car_controller = CarController(self.semantic_segmentation, debug_stream=debug_stream)

        # Setup pub/sub
        # WTF BUG!!! https://answers.ros.org/question/220502/image-subscriber-lag-despite-queue-1/
        self.rgb_camera_sub = rospy.Subscriber(config.TOPIC_GET_IMAGE, CompressedImage, callback=self.callback_rgb_image, queue_size=1, buff_size=2**24)
      

    def callback_detect_obstacle(self, data):

        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            image_np = cv2.resize(image_np, (320, 240))
            image_np = image_np[100:, :]
    
            self.danger_zone = self.depth_processor.pre_processing_depth_img(image_np)

        except CvBridgeError as e:
            print(e)


    def callback_rgb_image(self, data):
        '''
        Function to process rgb images
        '''
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # NOTE: image_np.shape = (240, 320, 3)
            image_np = cv2.resize(image_np, (320, 240))

            self.car_controller.control(image_np)

            if self.debug_stream:
                self.debug_stream.update_image('rgb', image_np)

        except CvBridgeError as e:
            print(e)