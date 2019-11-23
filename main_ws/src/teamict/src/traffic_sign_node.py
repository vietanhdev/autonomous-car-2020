#!/usr/bin/env python
from __future__ import print_function
import time
import config
import roslib
import rospy
from cv_bridge import CvBridge, CvBridgeError

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from src.sign_detection_faceboxes.sign_detector import SignDetector


# Only use CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Limit GPU usage
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000)])

if __name__ == '__main__':

    rospy.init_node(config.TEAM_NAME, anonymous=True)
    cv_bridge = CvBridge()

    image_processor = SignDetector(cv_bridge)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


