#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import time
import config
import roslib
import rospy
import cv2
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError
from param import Param
import geometry_msgs.msg
import threading

# Global variables for images
rgb_image = np.zeros((240, 320, 3), np.uint8)
rgb_image_mutex = threading.Lock()
depth_image = np.zeros((240, 320, 3), np.uint8)
depth_image_mutex = threading.Lock()
rgb_image = np.zeros((240, 320, 3), np.uint8)
rgb_image_mutex = threading.Lock()
depth_image = np.zeros((240, 320, 3), np.uint8)
depth_image_mutex = threading.Lock()

class ImageConverter:
    global rgb_image

    def __init__(self):
        '''
        Init ros node to receive images
        '''
        # ROS message exchange rate
        self.rate = rospy.Rate(10)
        self.bridge = CvBridge()

        # Setup pub/sub
        self.rgb_camera_sub = rospy.Subscriber(config.TOPIC_GET_IMAGE, CompressedImage, callback=self.callback_rgb_image, queue_size=1)
        self.depth_camera_sub = rospy.Subscriber(config.TOPIC_GET_DEPTH_IMAGE, CompressedImage, callback=self.callback_depth_image, queue_size=1)

        self.debug_image_pub = rospy.Publisher("/teamict/debug_image", Image, queue_size=1)


    def callback_rgb_image(self, data):
        '''
        Function to process rgb images
        '''
        global rgb_image, rgb_image_mutex
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # image_np = self.bridge.imgmsg_to_cv2(data)

            # NOTE: image_np.shape = (240, 320, 3)
            image_np = cv2.resize(image_np, (320, 240))

            # Copy to global variable
            rgb_image_mutex.acquire()
            rgb_image = image_np.copy()
            rgb_image_mutex.release()

        except CvBridgeError as e:
            print(e)


    def callback_depth_image(self, data):
        '''
        Function to process depth images
        '''
        global depth_image, depth_image_mutex
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # image_np = self.bridge.imgmsg_to_cv2(data)

            # NOTE: image_np.shape = (240, 320, 3)
            image_np = cv2.resize(image_np, (320, 240))
            
            # Copy to global variable
            depth_image_mutex.acquire()
            depth_image = image_np.copy()
            depth_image_mutex.release()

            # print(self.danger_zone)
        except CvBridgeError as e:
            print(e)


def image_viewer(bridge, debug_image_pub):
    global rgb_image, rgb_image_mutex
    global depth_image, depth_image_mutex

    _rgb_image = None
    _depth_image = None

    while not rospy.is_shutdown():
        
        time.sleep(0.1)

        # Copy images to local variables
        rgb_image_mutex.acquire()
        _rgb_image = rgb_image.copy()
        rgb_image_mutex.release()

        depth_image_mutex.acquire()
        _depth_image = depth_image.copy()
        depth_image_mutex.release()

        vis = np.concatenate((_rgb_image, _depth_image), axis=1)
        
        try:
            debug_image_pub.publish(bridge.cv2_to_imgmsg(vis, "bgr8"))
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    # cv2.namedWindow("Debug Image")

    rospy.init_node(config.TEAM_NAME, anonymous=True)
    ic = ImageConverter()

    # Start image viewer
    image_viewer_t = threading.Thread(target=image_viewer, args=(ic.bridge, ic.debug_image_pub))
    image_viewer_t.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        # image_viewer_t.join()
        # cv2.destroyAllWindows()


