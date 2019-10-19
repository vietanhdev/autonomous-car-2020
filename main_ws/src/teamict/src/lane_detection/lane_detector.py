#! /usr/bin/env python

"""
Lane Detection

    Usage: python3 run.py  --conf=./config.json

"""

import cv2
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
import numpy as np
import argparse
import json
import threading
import time
import rospy
import rospkg

class LaneDetector(threading.Thread):

    def __init__(self, config_path, model_path, image_queue, debug_stream=None):
        """
        Lane Detector
            @:param: config_path: path to configuration file
            @:param: model_path: path to model file
            @:param: image_queue: bgr image queue
            @:param: debug_stream: image stream for debugging
        """

        self.debug_stream = debug_stream
        self.image_queue = image_queue

        self.count = 0

        # Open and load configuration file
        with open(config_path) as config_buffer:
            self.config = json.loads(config_buffer.read())

        self.input_size = (self.config["model"]["im_width"], self.config["model"]["im_height"])

        # Load model
        self.model = tf.keras.models.load_model(model_path)

        # Initialize debug stream
        if self.debug_stream:
            self.debug_stream.create_stream('road_seg', 'debug/road_segmentation')
            
        threading.Thread.__init__(self)

    def run(self):
        while not rospy.is_shutdown():
            if not self.image_queue.empty():
                self.get_road_mask(self.image_queue.get())

                print(self.count)
                self.count += 1

    def get_road_mask(self, img):
        """
        Get road mask
        :param img: bgr image to get road mask
        :return: np.float32 binary image (1-road, 0-other)
        """

        img = cv2.resize(img, (self.input_size[0], self.input_size[1]))

        # Sub mean
        # Because we use it with the training samples, I put it here
        img = img.astype(np.float32)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
        img = img[ : , : , ::-1 ]

        # Pass through network
        net_input = np.expand_dims(img, axis=0)
        preds = self.model.predict(net_input, verbose=0)
        pred_1 = preds[:,:,:,1].reshape((self.input_size[1], self.input_size[0]))

        # Post processing
        road_mask = np.zeros((self.input_size[1], self.input_size[0]), np.uint8)
        road_mask[pred_1 >= self.config["road_threshold"]] = 255

        if self.debug_stream:
            self.debug_stream.update_image('road_seg', road_mask)

        return road_mask


    

