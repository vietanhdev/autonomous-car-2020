#!/usr/bin/env python
from __future__ import print_function
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

    debug_stream = DebugStream(cv_bridge)
    debug_stream.start()
    # debug_stream = None

    image_processor = ImageProcessor(cv_bridge, debug_stream)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        # cv2.destroyAllWindows()


