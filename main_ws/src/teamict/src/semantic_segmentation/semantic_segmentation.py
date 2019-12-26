#! /usr/bin/env python

"""
Semantic Segmentation module

    Usage: python3 run.py  --conf=./config.json

"""

import os
import cv2
import tensorflow as tf
import numpy as np
import time
import rospy
import rospkg
from .traffic_objects import TrafficObject, OBJECT_COLORS
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))
import config as gconfig
from .src.frontend import Segment

class SemanticSegmentation():

    def __init__(self, config_path, debug_stream=None):
        """
        Semantic Segmentation
            @:param: image_queue: bgr image queue
            @:param: debug_stream: image stream for debugging
        """

        self.debug_stream = debug_stream
        self.count = 0

        self.config = gconfig.SEMANTIC_SEGMENTATION_CONFIG
        self.input_size = (self.config["model"]["im_width"], self.config["model"]["im_height"])

        # define the model and train
        segment = Segment(self.config["model"]["backend"], self.input_size, self.config["model"]["nclasses"])
    
        self.model = segment.feature_extractor

        # Load best model
        self.model.load_weights(path.join(gconfig.DATA_FOLDER, self.config["model"]["model_file"]))

        # Load model
        # self.model = tf.keras.models.load_model(path.join(gconfig.DATA_FOLDER, self.config["model"]["model_file"]))

        # Initialize debug stream
        if self.debug_stream:
            self.debug_stream.create_stream('segmentation', 'debug/road_segmentation')


    def mask_with_color(self, img, mask, color=(255,255,255)):
        """
        Mask image with color
        """
        color_mask = np.zeros(img.shape, img.dtype)
        color_mask[:,:] = color
        color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)
        return cv2.addWeighted(color_mask, 0.5, img, 1, 0)


    def get_visualization_img(self, img, seg_masks):
        """
        Get visualized segmentation
        :param img: bgr image
        :return: bgr image masked with different colors corresponding to different classes
        """

        viz = img.copy()

        for tf_object in TrafficObject:
            viz = self.mask_with_color(viz, seg_masks[tf_object.name], OBJECT_COLORS[tf_object.name])

        return viz

    def get_masks(self, raw):
        """
        Get road mask
        :param img: bgr image to get road mask
        :return: np.float32 binary image (1-road, 0-other)
        """

        # Resize image to input shape
        raw = cv2.resize(raw, (self.input_size[0], self.input_size[1]))

        # Sub mean
        # Because we use it with the training samples, I put it here
        img = raw.astype(np.float32)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
        img = img[ : , : , ::-1 ]

        # Pass through network
        net_input = np.expand_dims(img, axis=0)
        preds = self.model.predict(net_input, verbose=0)

        # Extract masks
        seg_masks = {}
        for tf_object in TrafficObject:

            # Get predict value
            pred = preds[:, :, :, tf_object.value].reshape((self.input_size[1], self.input_size[0]))

            # Create uint8 thresh. mask
            mask = np.zeros((self.input_size[1], self.input_size[0]), np.uint8)
            mask[pred >= self.config["segmentation_thresh"][tf_object.name]] = 255

            # Append value
            seg_masks[tf_object.name] = mask


        # Create visualization if on debug
        if self.debug_stream:
            visualized_segmentation = self.get_visualization_img(raw, seg_masks)
            self.debug_stream.update_image('segmentation', visualized_segmentation)

        return seg_masks
