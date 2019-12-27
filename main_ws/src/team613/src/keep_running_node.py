#!/usr/bin/env python

'''
This node publish a speed = 0 when not hearing from speed topic for a duration.
This trick prevent simulator from not sending image to our control node
'''

from __future__ import print_function
import time
import config
import roslib
import rospy
from std_msgs.msg import Float32


last_time_set_speed = 0
def set_speed_received(data):
    global last_time_set_speed
    last_time_set_speed = time.time()

if __name__ == '__main__':

    rospy.init_node(config.TEAM_NAME, anonymous=True)
    
    # Last time publishing speed and angle
    # Save to restart transfering images
    speed_pub = rospy.Publisher(config.TOPIC_SET_SPEED, Float32, queue_size=1)

    rospy.Subscriber(config.TOPIC_SET_SPEED, Float32, set_speed_received, queue_size=1)

    r = rospy.Rate(0.5)

    while not rospy.is_shutdown():
        if time.time() - last_time_set_speed > 2:
            speed_pub.publish(0)
        r.sleep()


