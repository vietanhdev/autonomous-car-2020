#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import time
import config
import roslib
import rospy
import rospkg
import cv2 as cv
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

        self.trafficsign_pub = rospy.Publisher('/team613/trafficsign', Int32, queue_size=3)
    

    def callback_depth_image(self, data):
        '''
        Function to process depth images
        '''
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv.imdecode(np_arr, cv.IMREAD_GRAYSCALE)
            image_np = cv.resize(image_np, (320, 240))

            self.callback_processing_thread(image_np)

        except CvBridgeError as e:
            print(e)


    def callback_rgb_image(self, data):
        '''
        Function to process rgb images
        '''
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv.imdecode(np_arr, cv.IMREAD_COLOR)

            # NOTE: image_np.shape = (240, 320, 3)
            image_np = cv.resize(image_np, (320, 240))

            self.image_mutex.acquire()
            self.image = image_np.copy()
            self.image_mutex.release()

        except CvBridgeError as e:
            print(e)


    # https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/
    def detect_keypoints(self, depth_img):

        # Filter far objects
        depth_img[depth_img > 170] = 0

        depth_img = cv.medianBlur(depth_img, 5)

        # Filter small objects
        kernel = np.ones((5,5),np.uint8)
        gray = cv.morphologyEx(depth_img, cv.MORPH_OPEN, kernel, iterations=1)
        
        # Set our filtering parameters 
        # Initialize parameter settiing using cv.SimpleBlobDetector 
        params = cv.SimpleBlobDetector_Params() 

        # Set Area filtering parameters 
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 400

        params.filterByColor = False

        # Set Circularity filtering parameters 
        params.filterByCircularity = True 
        params.minCircularity = 0.5

        # Set Convexity filtering parameters 
        params.filterByConvexity = True
        params.minConvexity = 0.2
            
        # Set inertia filtering parameters 
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        
        # Create a detector with the parameters 
        detector = cv.SimpleBlobDetector_create(params) 
            
        # Detect blobs 
        keypoints = detector.detect(depth_img)

        return keypoints


    def callback_processing_thread(self, depth_image):
        '''
        Processing thread
        '''

        # Local copy
        image = None
        self.image_mutex.acquire()
        if self.image is not None:
            image = self.image.copy()
        self.image_mutex.release()

        if image is None:
            return


        # ======= Process depth image =======
        keypoints = self.detect_keypoints(depth_image)

        # Draw 
        if gconfig.DEVELOPMENT:
            draw = image.copy()
            blank = np.zeros((1, 1))  
            draw = cv.drawKeypoints(draw, keypoints, blank, (0, 255, 0), 
                                    cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 

        rects = []
        crops = []


        img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        for keypoint in keypoints:
            x = keypoint.pt[0]
            y = keypoint.pt[1]
            center = (int(x), int(y))
            radius = int(keypoint.size / 2) 
            
            # Bounding box:
            im_height, im_width, _ = img_rgb.shape
            pad = int(0.4*radius)
            tl_x = max(0, center[0] - radius - pad)
            tl_y = max(0, center[1] - radius - pad)
            br_x = min(im_width-1, tl_x + 2 * radius + pad)
            br_y = min(im_height-1, tl_y + 2 * radius + pad)

            # cv.rectangle(draw, (tl_x, tl_y), (br_x, br_y), (200, 200, 200), 2)
            rect = ((tl_x, tl_y), (br_x, br_y))

            crop = img_rgb[tl_y:br_y, tl_x:br_x]

            if crop.shape[0] > 0 and crop.shape[1] > 0:
                crop = cv.resize(crop, (32, 32))
                crop = crop.astype(np.float32) / 255.0
                crops.append(crop)
                rects.append(rect)

        if len(crops) != 0:
                
            preds = self.sign_model.predict(np.array(crops))
            preds = np.argmax(preds, axis=1)
            preds = preds.tolist()


            for i in range(len(preds)):
                if preds[i] == 0:

                    if gconfig.DEVELOPMENT:
                        cv.rectangle(draw, rects[i][0], rects[i][1], (0, 0, 255), 3)
                        draw = cv.putText(draw, "LEFT", rects[i][0], cv.FONT_HERSHEY_SIMPLEX ,  
                            0.5, (0,255,0), 1, cv.LINE_AA) 

                    self.trafficsign_pub.publish(Int32(0))
                    print("LEFT")
                elif preds[i] == 1:

                    if gconfig.DEVELOPMENT:
                        cv.rectangle(draw, rects[i][0], rects[i][1], (255, 0, 0), 3)
                        draw = cv.putText(draw, "RIGHT", rects[i][0], cv.FONT_HERSHEY_SIMPLEX ,  
                            0.5, (0,255,0), 1, cv.LINE_AA) 

                    self.trafficsign_pub.publish(Int32(1))
                    print("RIGHT")

            if gconfig.DEVELOPMENT:
                self.sign_debug_stream_pub.publish(self.cv_bridge.cv2_to_imgmsg(draw, "bgr8"))