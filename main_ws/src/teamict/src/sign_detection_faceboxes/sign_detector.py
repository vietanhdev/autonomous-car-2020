#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import time
import roslib
import rospy
import rospkg
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
import threading
from threading import Thread, Lock
import rospy
import tensorflow as tf
from std_msgs.msg import Int32
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))
import config as gconfig
from lib.core.api.face_detector import FaceDetector
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections import deque

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/epoch_1_val_loss0.298565")
detector = FaceDetector(model_path)

class SignDetector:

    def __init__(self, cv_bridge):
        '''
        Init ros node to receive images
        '''

        self.cv_bridge = cv_bridge

        self.image_mutex = Lock()
        self.image = None

        self.rate = rospy.Rate(50) 

        self.sign_model = tf.keras.models.load_model(path.join(gconfig.DATA_FOLDER, gconfig.SIGN_DETECTION_CONFIG['model_file']))

        if gconfig.DEVELOPMENT:
            self.sign_debug_stream_pub = rospy.Publisher("debug/sign_detection", Image, queue_size=1)

        # Setup pub/sub
        # WTF BUG!!! https://answers.ros.org/question/220502/image-subscriber-lag-despite-queue-1/
        self.rgb_camera_sub = rospy.Subscriber(gconfig.TOPIC_GET_IMAGE, CompressedImage, callback=self.callback_rgb_image, queue_size=1, buff_size=2**24)
        self.depth_camera_sub = rospy.Subscriber(gconfig.TOPIC_GET_DEPTH_IMAGE, CompressedImage, callback=self.callback_depth_image, queue_size=1, buff_size=2**8)

        self.trafficsign_pub = rospy.Publisher('/teamict/trafficsign', Int32, queue_size=3)

        self.traffic_sign_queue = deque(maxlen=8) # Traffic sign will be stored as (<sign>, <time>)

    def callback_depth_image(self, data):
        '''
        Function to process depth images
        '''
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            image_np = cv2.resize(image_np, (320, 240))

            self.callback_processing_thread(image_np)

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

            self.image_mutex.acquire()
            self.image = image_np.copy()
            self.image_mutex.release()

        except CvBridgeError as e:
            print(e)

    def remove_expired_signs(self):
        current_time = time.time()
        while len(self.traffic_sign_queue) > 0:
            oldest_sign, detected_time = self.traffic_sign_queue[0]
            if detected_time < current_time - gconfig.SIGN_EXP_TIME: # Remove a expired sign
                self.traffic_sign_queue.popleft()
            else:
                break


    def get_traffic_sign_filtered(self):

        # Filter all expired traffic sign
        # NOTE!!! If trafficsign number is reduced from a positive number
        # to zero, we publish a NO_SIGN signal to reset traffic sign status
        n_traffic_signs_before_remove_expired = len(self.traffic_sign_queue)
        self.remove_expired_signs()
        n_traffic_signs_after_remove_expired = len(self.traffic_sign_queue)
        if n_traffic_signs_before_remove_expired > 0 and n_traffic_signs_after_remove_expired == 0:
            self.trafficsign_pub.publish(Int32(gconfig.SIGN_NO_SIGN))
            self.trafficsign_pub.publish(Int32(gconfig.SIGN_NO_SIGN))

        # Count
        n_turn_left = 0
        n_turn_right = 0

        for i in range(len(self.traffic_sign_queue)):
            sign, detected_time = self.traffic_sign_queue[i]
            if sign == gconfig.SIGN_LEFT:
                n_turn_left += 1
            else:
                n_turn_right += 1

        if n_turn_left > n_turn_right and n_turn_left >= gconfig.SIGN_DETECTION_THRESHOLD: # Turn left
            self.trafficsign_pub.publish(Int32(gconfig.SIGN_LEFT))
            self.traffic_sign_queue.clear() # Clear queue after conslusion
        elif n_turn_left < n_turn_right and n_turn_right > gconfig.SIGN_DETECTION_THRESHOLD: # Turn right
            self.trafficsign_pub.publish(Int32(gconfig.SIGN_RIGHT))
            self.traffic_sign_queue.clear() # Clear queue after conslusion

    def callback_processing_thread(self, depth_image):
        '''
        Processing thread
        '''

        global detector

        # Local copy
        image = None
        self.image_mutex.acquire()
        if self.image is not None:
            image = self.image.copy()
        self.image_mutex.release()

        if image is None:
            return


        # ======= Process depth image =======
        
        draw = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pred = detector(image, 0.4)

        for i, pred_i in enumerate(pred):
            if pred_i.shape[0] > 0:
                for bbox in pred_i:

                    tl_x = bbox[0]
                    tl_y = bbox[1]
                    br_x = bbox[2]
                    br_y = bbox[3]

                    if i == 0:
                        sign_label = "RIGHT"
                        self.traffic_sign_queue.append((gconfig.SIGN_RIGHT, time.time()))
                    else:
                        sign_label = "LEFT"
                        self.traffic_sign_queue.append((gconfig.SIGN_LEFT, time.time()))

                    if gconfig.DEVELOPMENT:
                        cv2.rectangle(draw, (tl_x, tl_y), (br_x, br_y), (0, 0, 255), 3)
                        draw = cv2.putText(draw, sign_label, (tl_x, tl_y), cv2.FONT_HERSHEY_SIMPLEX ,  
                            0.5, (0,255,0), 1, cv2.LINE_AA)

        traffic_sign = self.get_traffic_sign_filtered()
        
        if gconfig.DEVELOPMENT:
            self.sign_debug_stream_pub.publish(self.cv_bridge.cv2_to_imgmsg(draw, "bgr8"))
            