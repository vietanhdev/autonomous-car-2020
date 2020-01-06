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

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from src.sign_detection_faceboxes.sign_detector import SignDetector

import Queue
from threading import Thread


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

        self.image_queue = Queue.Queue(maxsize=config.IMAGE_QUEUE_SIZE)
        self.depth_image_queue = Queue.Queue(maxsize=config.IMAGE_QUEUE_SIZE)

        # Segmentation
        seg_config = path.join(DATA_FOLDER, 'semantic_seg_ENet.conf.json')
        self.semantic_segmentation = SemanticSegmentation(seg_config, debug_stream=debug_stream)

        # Depth processor
        self.depth_processor = DepthProcessor(debug_stream=debug_stream)

        # Sign detector
        self.image_processor = SignDetector(cv_bridge)

        # Car controlling
        self.car_controller = CarController(self.semantic_segmentation, debug_stream=debug_stream)

        self.sign_detector = SignDetector(cv_bridge)

        # Setup pub/sub
        # WTF BUG!!! https://answers.ros.org/question/220502/image-subscriber-lag-despite-queue-1/
        self.rgb_camera_sub = rospy.Subscriber(config.TOPIC_GET_IMAGE, CompressedImage, callback=self.callback_rgb_image, queue_size=1, buff_size=2**24)

        if config.USE_DEPTH_CAMERA:
            self.depth_camera_sub = rospy.Subscriber(config.TOPIC_GET_DEPTH_IMAGE, CompressedImage, callback=self.callback_depth_image, queue_size=1, buff_size=2**24)


        # Setup workers
        control_worker = Thread(target=self.thread_car_control, args=(self.image_queue, self.depth_image_queue))
        control_worker.setDaemon(True)
        control_worker.start()

        sign_worker = Thread(target=self.sign_detector.processing_thread, args=(self.image_queue,))
        sign_worker.setDaemon(True)
        sign_worker.start()

        connection_worker = Thread(target=self.thread_refresh_connection)
        connection_worker.setDaemon(True)
        connection_worker.start()
        self.last_time_image_received = time.time()


        # Push init images to force model loading
        image_np = np.zeros([240,320,3],dtype=np.uint8)
        depth_image_np = np.zeros([240,320],dtype=np.uint8)
        try:
            for _ in range(10):
                if self.image_queue.full():
                    break
                self.image_queue.put_nowait(image_np)
            for _ in range(10):
                if self.depth_image_queue.full():
                    break
                self.depth_image_queue.put_nowait(depth_image_np)
        except:
            pass


    def thread_refresh_connection(self):
        '''
        Monitor and restart image topic if we don't receive from image topic for a long time
        This prevent car freezing when start simulator after start controling node.
        This idea comes from https://github.com/RobotWebTools/rosbridge_suite/issues/298.
        '''
        while True:
            time.sleep(3)
            if time.time() - self.last_time_image_received > 3:
                self.rgb_camera_sub.unregister()
                # WTF BUG!!! https://answers.ros.org/question/220502/image-subscriber-lag-despite-queue-1/
                self.rgb_camera_sub = rospy.Subscriber(config.TOPIC_GET_IMAGE, CompressedImage, callback=self.callback_rgb_image, queue_size=1, buff_size=2**24)

                if config.USE_DEPTH_CAMERA:
                    self.depth_camera_sub.unregister()
                    self.depth_camera_sub = rospy.Subscriber(config.TOPIC_GET_DEPTH_IMAGE, CompressedImage, callback=self.callback_depth_image, queue_size=1, buff_size=2**24)


    def thread_car_control(self, image_queue, depth_image_queue):
        while True:
            img = image_queue.get()

            if config.USE_DEPTH_CAMERA:
                depth_img = depth_image_queue.get()
                self.car_controller.control(img, depth_img)
            else:
                self.car_controller.control(img)


    def callback_rgb_image(self, data):
        '''
        Function to process rgb images
        '''
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # NOTE: image_np.shape = (240, 320, 3)
            image_np = cv2.resize(image_np, (320, 240))


            for _ in range(2):

                if self.image_queue.full():
                    try:
                        self.image_queue.get_nowait()
                    except e:
                        print(e)

                try:
                    self.image_queue.put_nowait(image_np)
                except e:
                    print(e)

            # self.car_controller.control(image_np)

            self.last_time_image_received = time.time()

            if self.debug_stream:
                self.debug_stream.update_image('rgb', image_np)
                

        except CvBridgeError as e:
            print(e)

    
    def callback_depth_image(self, data):

        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            image_np = cv2.resize(image_np, (320, 240))
    
            image = self.depth_processor.pre_processing_depth_img(image_np)

            for _ in range(2):

                if self.depth_image_queue.full():
                    try:
                        self.depth_image_queue.get_nowait()
                    except e:
                        print(e)

                try:
                    self.depth_image_queue.put_nowait(image)
                except e:
                    print(e)

        except CvBridgeError as e:
            print(e)