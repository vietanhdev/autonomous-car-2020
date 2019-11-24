#!/usr/bin/env python
from __future__ import print_function
import time
import config
import roslib
import rospy
from std_msgs.msg import Float32


if __name__ == '__main__':

    rospy.init_node(config.TEAM_NAME, anonymous=True)
    
    # Last time publishing speed and angle
    # Save to restart transfering images
    speed_pub = rospy.Publisher(config.TOPIC_SET_SPEED, Float32, queue_size=1)

    r = rospy.Rate(0.2) #10hz

    while not rospy.is_shutdown():
        speed_pub.publish(0)
        r.sleep()


