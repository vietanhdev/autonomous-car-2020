#!/usr/bin/env python

from __future__ import print_function
# Limit GPU usage
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2200)])
import time
import config
import roslib
import rospy
from cv_bridge import CvBridge, CvBridgeError
from debug_stream import DebugStream
from image_processor import ImageProcessor



if __name__ == '__main__':

    rospy.init_node(config.TEAM_NAME, anonymous=True)
    cv_bridge = CvBridge()

    if config.DEVELOPMENT:
        debug_stream = DebugStream(cv_bridge)
        debug_stream.start()
    else:
        debug_stream = None

    image_processor = ImageProcessor(cv_bridge, debug_stream)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


