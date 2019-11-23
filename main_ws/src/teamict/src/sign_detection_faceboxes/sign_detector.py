#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import time
import config
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

        self.sign_model = tf.keras.models.load_model(path.join(config.DATA_FOLDER, config.SIGN_DETECTION_CONFIG['model_file']))

        if gconfig.DEVELOPMENT:
            self.sign_debug_stream_pub = rospy.Publisher("debug/sign_detection", Image, queue_size=1)

        # Setup pub/sub
        # WTF BUG!!! https://answers.ros.org/question/220502/image-subscriber-lag-despite-queue-1/
        self.rgb_camera_sub = rospy.Subscriber(gconfig.TOPIC_GET_IMAGE, CompressedImage, callback=self.callback_rgb_image, queue_size=1, buff_size=2**24)
        self.depth_camera_sub = rospy.Subscriber(gconfig.TOPIC_GET_DEPTH_IMAGE, CompressedImage, callback=self.callback_depth_image, queue_size=1, buff_size=2**8)

        self.trafficsign_pub = rospy.Publisher('/teamict/trafficsign', Int32, queue_size=3)

        self.last_traffic_sign_time = 0

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


        # ======= Notify no traffic sign when traffic sign timeout =======
        if self.last_traffic_sign_time + 10 < time.time() and self.last_traffic_sign_time + 100 > time.time():
            self.last_traffic_sign_time = time.time()
            self.trafficsign_pub.publish(Int32(0))


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

                    print(bbox)

                    if i == 0:
                        sign_label = "RIGHT"
                        self.trafficsign_pub.publish(Int32(1))
                    else:
                        sign_label = "LEFT"
                        self.trafficsign_pub.publish(Int32(-1))

                    if gconfig.DEVELOPMENT:
                        cv2.rectangle(draw, (tl_x, tl_y), (br_x, br_y), (0, 0, 255), 3)
                        draw = cv2.putText(draw, sign_label, (tl_x, tl_y), cv2.FONT_HERSHEY_SIMPLEX ,  
                            0.5, (0,255,0), 1, cv2.LINE_AA) 

                    self.last_traffic_sign_time = time.time()
        
        if gconfig.DEVELOPMENT:
            self.sign_debug_stream_pub.publish(self.cv_bridge.cv2_to_imgmsg(draw, "bgr8"))
            